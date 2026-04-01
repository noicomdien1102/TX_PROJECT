import json
import math
import os
import re
import datetime
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── App Setup ──────────────────────────────────────────────────────────────

app = FastAPI(title="TX Prediction Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
if os.environ.get("VERCEL") == "1":
    DATA_DIR = "/tmp/data"
os.makedirs(DATA_DIR, exist_ok=True)

# ─── Data Models ────────────────────────────────────────────────────────────

class DiceInput(BaseModel):
    d1: int
    d2: int
    d3: int
    session_id: str = "default"

class ManualInput(BaseModel):
    result: str
    session_id: str = "default"

class ConfirmInput(BaseModel):
    actual: str           # "T" or "X"
    session_id: str = "default"

class CreateSessionInput(BaseModel):
    session_id: str
    description: str = ""

# ─── Session I/O ────────────────────────────────────────────────────────────

def _safe_session_id(session_id: str) -> str:
    """Sanitise to filesystem-safe characters."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)[:64]


def _session_path(session_id: str) -> str:
    safe = _safe_session_id(session_id)
    return os.path.join(DATA_DIR, f"session_{safe}.json")


def load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return {
            "session_id": session_id,
            "description": "",
            "history": [],
            "pattern_weights": {},
        }
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {
                "session_id": session_id,
                "description": "",
                "history": [],
                "pattern_weights": {},
            }


def save_session(session: dict):
    path = _session_path(session["session_id"])
    with open(path, "w") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)


# ─── History Schema Helpers ──────────────────────────────────────────────────

def _make_prediction_snapshot(history: list[dict], sequence: list[str], pattern_weights: dict) -> dict:
    """Compute the prediction snapshot before appending the new entry."""
    if len(sequence) < 3:
        return {
            "suggest": None,
            "prob_t": None,
            "prob_x": None,
            "top_pattern": None,
            "confirmed": False,
            "was_correct": None,
        }
    probs = advanced_predict(sequence)
    top_patterns = detect_top_patterns(sequence, pattern_weights)
    bias = detect_bias(history)
    blend_w = compute_blend_weights(sequence, top_patterns)

    if top_patterns:
        # Rank-weighted blend: #1 pattern gets 60% say, #2 gets 25%, #3 gets 15%.
        # This ensures the top (most confident) pattern is decisive even when
        # multiple lower-ranked patterns oppose it from the other side.
        _RANK_W = [0.60, 0.25, 0.15]
        t_mass = sum(_RANK_W[i] for i, p in enumerate(top_patterns[:3]) if p["expected"] == "T")
        x_mass = sum(_RANK_W[i] for i, p in enumerate(top_patterns[:3]) if p["expected"] == "X")
        _total = t_mass + x_mass
        if _total > 0:
            t_mass /= _total
            x_mass /= _total
        pw = blend_w["pattern"]
        mw = blend_w["markov"]
        probs["T"] = round(probs["T"] * mw + t_mass * pw, 4)
        probs["X"] = round(probs["X"] * mw + x_mass * pw, 4)

    suggest = apply_bias_influence(probs, bias, history)
    top_pattern = top_patterns[0] if top_patterns else None

    return {
        "suggest": suggest,
        "prob_t": probs.get("T"),
        "prob_x": probs.get("X"),
        "top_pattern": top_pattern,
        "confirmed": False,
        "was_correct": None,
    }


def _make_entry(result: str, source: str, dice: Optional[list], prediction: dict) -> dict:
    return {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "result": result,
        "source": source,
        "dice": dice,
        "prediction": prediction,
    }


# ─── Accuracy ───────────────────────────────────────────────────────────────

def compute_accuracy(history: list[dict]) -> dict:
    confirmed = [h for h in history if h.get("prediction") and h["prediction"].get("confirmed")]
    total = len(confirmed)
    if total == 0:
        return {"total_confirmed": 0, "accuracy_last_20": None, "accuracy_last_50": None,
                "accuracy_last_100": None, "accuracy_all": None}

    def _acc(entries):
        if not entries:
            return None
        correct = sum(1 for e in entries if e["prediction"].get("was_correct"))
        return round(correct / len(entries) * 100, 2)

    return {
        "total_confirmed": total,
        "accuracy_last_20": _acc(confirmed[-20:]),
        "accuracy_last_50": _acc(confirmed[-50:]),
        "accuracy_last_100": _acc(confirmed[-100:]),
        "accuracy_all": _acc(confirmed),
    }


# ─── Dynamic Blend Weights ───────────────────────────────────────────────────

def compute_blend_weights(sequence: list[str], top_patterns: list[dict]) -> dict:
    n = len(sequence)
    if n < 20:
        base = 0.20
    elif n < 50:
        base = 0.40
    elif n < 100:
        base = 0.55
    else:
        base = 0.65

    if not top_patterns:
        base -= 0.15
    else:
        top_prob = top_patterns[0]["prob"]
        if top_prob >= 0.70:
            base += 0.15
        elif top_prob >= 0.60:
            base += 0.08

    pattern_w = max(0.20, min(0.80, base))
    markov_w = round(1.0 - pattern_w, 4)
    return {"markov": round(markov_w, 4), "pattern": round(pattern_w, 4)}


# ─── Image Processing ───────────────────────────────────────────────────────

def process_grid_image(file_bytes: bytes) -> tuple[list[str], list[dict], list[dict]]:
    """Decode image, auto-crop padding, resize to 1000x250, split into 20x5 grid.
    Read in snake pattern. Safely detect T (Black) or X (White) by darkest pixels.
    Returns (results, debug_log, low_confidence_cells)."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot decode image")

    # 1. Auto-Crop padding
    _, thresh_bright = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(img, 50, 150)
    mask = cv2.bitwise_or(thresh_bright, edges)

    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) > 0 and len(x_coords) > 0:
        y1_crop = max(0, np.min(y_coords) - 5)
        y2_crop = min(img.shape[0], np.max(y_coords) + 5)
        x1_crop = max(0, np.min(x_coords) - 5)
        x2_crop = min(img.shape[1], np.max(x_coords) + 5)
        img = img[y1_crop:y2_crop, x1_crop:x2_crop]

    # 2. Extract cells
    img_resized = cv2.resize(img, (1000, 250), interpolation=cv2.INTER_AREA)

    cells_data = []
    for col in range(20):
        rows = range(5) if col % 2 == 0 else range(4, -1, -1)
        for row in rows:
            y1 = row * 50 + 10
            y2 = row * 50 + 40
            x1 = col * 50 + 10
            x2 = col * 50 + 40
            cell_core = img_resized[y1:y2, x1:x2]
            sorted_pixels = np.sort(cell_core.flatten())
            dark_mean = float(np.mean(sorted_pixels[:100]))
            cells_data.append({"col": col, "row": row, "dark_mean": dark_mean})

    # Dynamic Thresholding
    all_dark_means = [c["dark_mean"] for c in cells_data]
    min_p5 = float(np.percentile(all_dark_means, 5))
    max_p95 = float(np.percentile(all_dark_means, 95))
    threshold = (min_p5 + max_p95) / 2

    # Image confidence margin (change 7 – low-confidence cells)
    margin = (max_p95 - min_p5) * 0.15

    results: list[str] = []
    debug_log: list[dict] = []
    low_confidence_cells: list[dict] = []

    for c in cells_data:
        res_val = "T" if c["dark_mean"] < threshold else "X"
        results.append(res_val)
        is_low = abs(c["dark_mean"] - threshold) < margin

        entry = {
            "time_idx": len(results) - 1,
            "col": c["col"],
            "row": c["row"],
            "dark_mean": round(c["dark_mean"], 1),
            "result": res_val,
            "confidence": "low" if is_low else "ok",
        }
        debug_log.append(entry)
        if is_low:
            low_confidence_cells.append(entry)

    return results, debug_log, low_confidence_cells


# ─── Dice Logic ─────────────────────────────────────────────────────────────

def dice_to_result(d1: int, d2: int, d3: int) -> str:
    total = d1 + d2 + d3
    return "T" if total >= 11 else "X"


# ─── Advanced Run-Length Predictor ──────────────────────────────────────────

def advanced_predict(sequence: list[str]) -> dict:
    """Predict based on Run-Length Continuation historical frequency + Momentum."""
    if len(sequence) < 3:
        return {"T": 0.5, "X": 0.5}

    current_val = sequence[-1]
    other_val = "X" if current_val == "T" else "T"

    streak_len = 1
    for v in reversed(sequence[:-1]):
        if v == current_val:
            streak_len += 1
        else:
            break

    hist_runs = []
    cur, cnt = sequence[0], 1
    for v in sequence[1:]:
        if v == cur:
            cnt += 1
        else:
            if cur == current_val:
                hist_runs.append(cnt)
            cur, cnt = v, 1

    reached_len = [r for r in hist_runs if r >= streak_len]
    continued = sum(1 for r in reached_len if r > streak_len)
    broken = sum(1 for r in reached_len if r == streak_len)

    prob_cont = (continued + 1.0) / (continued + broken + 2.0)
    prob_broke = (broken + 1.0) / (continued + broken + 2.0)

    recent = sequence[-20:] if len(sequence) >= 20 else sequence
    recent_current_val_ratio = recent.count(current_val) / len(recent)
    recent_other_val_ratio = recent.count(other_val) / len(recent)

    final_prob_cont = prob_cont * 0.7 + recent_current_val_ratio * 0.3
    final_prob_broke = prob_broke * 0.7 + recent_other_val_ratio * 0.3

    total = final_prob_cont + final_prob_broke
    p_current = final_prob_cont / total
    p_other = final_prob_broke / total

    return {
        current_val: round(p_current, 4),
        other_val: round(p_other, 4)
    }


# ─── Pattern Probability Engine ─────────────────────────────────────────────

def _score_pattern_tail_match(pattern_name: str, r: list[int], current_val: str) -> tuple[float, str, str]:
    """Evaluate how well the current tail matches the pattern and returning (Multiplier, ExpectedNext, Detail)."""
    switch = "X" if current_val == "T" else "T"
    cont = current_val

    if len(r) == 0:
        return 0.0, cont, ""

    if pattern_name == "Cầu 1-1 (Ping Pong)":
        if len(r) >= 6 and all(x == 1 for x in r[-6:]): return 6.0, switch, "Ping pong quá dài (Chuẩn bị gãy)"
        if len(r) >= 3 and r[-1] == 1 and r[-2] == 1 and r[-3] == 1: return 10.0, switch, "Đang xen kẽ T/X liên tục"
        if len(r) >= 2 and r[-1] == 1 and r[-2] == 1: return 3.0, switch, "Dấu hiệu Cầu 1-1"
        return 0.0, switch, ""

    if pattern_name == "Cầu 2-2 (Nhịp đôi)":
        if len(r) >= 2 and r[-2] == 2 and r[-1] == 2: return 8.0, switch, "Nhịp đôi đều đặn"
        if len(r) >= 3 and r[-3] == 2 and r[-2] == 2 and r[-1] == 1: return 10.0, cont, "Chờ hoàn thành nhịp 2"
        # Raised from 4.0 → 7.0 so 2-2 beats Cầu 2-1 (5.0) when both fire on tail [2,1]
        if len(r) >= 2 and r[-2] == 2 and r[-1] == 1: return 7.0, cont, "Chờ hoàn thành nhịp 2"
        return 0.0, cont, ""

    if pattern_name == "Cầu 2-1-2-1 (Lặp lệch)":
        if len(r) >= 3 and r[-3] == 2 and r[-2] == 1 and r[-1] == 2: return 10.0, switch, "Vừa nhịp 2, chờ 1"
        if len(r) >= 4 and r[-4] == 2 and r[-3] == 1 and r[-2] == 2 and r[-1] == 1: return 9.0, cont, "Vừa nhịp 1, chờ 2"
        return 0.0, cont, ""

    if pattern_name == "Cầu Tăng Dần (1-2-3)":
        if len(r) >= 2 and r[-2] == 1 and r[-1] == 2: return 6.0, switch, "Nhịp 1-2 xong, xả bẻ sang 3"
        if len(r) >= 3 and r[-3] == 1 and r[-2] == 2 and r[-1] == 1: return 8.0, cont, "Đang lên nhịp 3"
        if len(r) >= 3 and r[-3] == 1 and r[-2] == 2 and r[-1] == 2: return 9.0, cont, "Đang lên nhịp 3"
        if len(r) >= 3 and r[-3] == 1 and r[-2] == 2 and r[-1] == 3: return 8.0, switch, "Hoàn thành 1-2-3, xả cầu"
        return 0.0, cont, ""

    if pattern_name == "Cầu Giảm Dần (3-2-1)":
        if len(r) >= 2 and r[-2] == 3 and r[-1] == 2: return 8.0, switch, "Nhịp 3-2 xong, bắt nhịp 1"
        if len(r) >= 3 and r[-3] == 3 and r[-2] == 2 and r[-1] == 1: return 8.0, switch, "Hoàn thành 3-2-1, xả cầu"
        return 0.0, cont, ""

    if pattern_name == "Cầu Bẻ Giả (Fake Break)":
        if len(r) >= 3 and r[-3] >= 3 and r[-2] == 1 and r[-1] == 1: return 8.0, cont, "Dấu hiệu bẻ giả, trở lại Bệt"
        if len(r) >= 3 and r[-3] >= 3 and r[-2] == 1 and r[-1] >= 2: return 10.0, cont, "Bẻ giả thành công, tiếp tục Bệt"
        return 0.0, cont, ""

    if pattern_name == "Cầu 2-4 / 4-2":
        if len(r) >= 2 and r[-2] == 2 and r[-1] == 3: return 8.0, cont, "Đang dồn nhịp 4"
        if len(r) >= 2 and r[-2] == 2 and r[-1] == 4: return 8.0, switch, "Hoàn thành 2-4"
        if len(r) >= 2 and r[-2] == 4 and r[-1] == 1: return 8.0, cont, "Đang xuống nhịp 2"
        if len(r) >= 2 and r[-2] == 4 and r[-1] == 2: return 8.0, switch, "Hoàn thành 4-2"
        return 0.0, cont, ""

    if pattern_name == "Cầu 3-4 / 4-3":
        if len(r) >= 2 and r[-2] == 3 and r[-1] == 3: return 8.0, cont, "Đang dồn nhịp 4"
        if len(r) >= 2 and r[-2] == 3 and r[-1] == 4: return 8.0, switch, "Hoàn thành 3-4"
        if len(r) >= 2 and r[-2] == 4 and r[-1] == 2: return 8.0, cont, "Đang xuống nhịp 3"
        if len(r) >= 2 and r[-2] == 4 and r[-1] == 3: return 8.0, switch, "Hoàn thành 4-3"
        return 0.0, cont, ""

    if pattern_name == "Cầu 2-5-2-4":
        if len(r) >= 4 and r[-4] == 2 and r[-3] == 5 and r[-2] == 2 and r[-1] <= 3: return 12.0, cont, "Chờ nhịp 4 cuối của 2-5-2-4"
        if len(r) >= 4 and r[-4] == 2 and r[-3] == 5 and r[-2] == 2 and r[-1] == 4: return 10.0, switch, "Hoàn thành 2-5-2-4"
        return 0.0, cont, ""

    if pattern_name == "Cầu Double Bệt (Bệt Đối Xứng)":
        if len(r) >= 2 and r[-2] >= 3 and r[-1] >= 2 and r[-1] < r[-2]: return 6.0, cont, "Cân bệt trước đó"
        if len(r) >= 2 and r[-2] >= 3 and r[-1] == r[-2]: return 8.0, switch, "Đã cân xong 2 bệt"
        return 0.0, cont, ""

    if pattern_name == "Cầu 3-1":
        if len(r) >= 2 and r[-2] == 3 and r[-1] == 1: return 8.0, switch, "Xong nhịp 3-1"
        if len(r) >= 3 and r[-3] == 3 and r[-2] == 1 and r[-1] <= 2: return 8.0, cont, "Chờ nhịp 3"
        return 0.0, cont, ""
    if pattern_name == "Cầu 1-3":
        if len(r) >= 2 and r[-2] == 1 and r[-1] == 3: return 8.0, switch, "Xong nhịp 1-3"
        if len(r) >= 2 and r[-2] == 1 and r[-1] <= 2: return 6.0, cont, "Chờ nhịp 3"
        return 0.0, cont, ""
    if pattern_name == "Cầu 1-2" or pattern_name == "Cầu 2-1":
        if len(r) >= 2 and r[-2] == 1 and r[-1] == 2: return 5.0, switch, "Nhịp 1-2"
        if len(r) >= 2 and r[-2] == 2 and r[-1] == 1: return 5.0, switch, "Nhịp 2-1"
        return 0.0, cont, ""

    return 0.0, cont, ""


def detect_top_patterns(sequence: list[str], pattern_weights: Optional[dict] = None) -> list[dict]:
    """Return top 3 pattern probabilities based on history freq + tail match + equilibrium.
    Uses sequence[-30:] for T/X ratio check (rolling window equilibrium)."""
    if len(sequence) < 5:
        return []

    if pattern_weights is None:
        pattern_weights = {}

    # Change 6 – Rolling window equilibrium: use last 30 entries for ratio check
    window = sequence[-30:]
    t_count = window.count("T")
    x_count = window.count("X")
    t_ratio = t_count / len(window)
    x_ratio = x_count / len(window)

    # Convert history into runs
    runs: list[tuple[str, int]] = []
    cur, cnt = sequence[0], 1
    for v in sequence[1:]:
        if v == cur:
            cnt += 1
        else:
            runs.append((cur, cnt))
            cur, cnt = v, 1
    runs.append((cur, cnt))
    run_lengths = [r[1] for r in runs]

    hist_freq = {
        "Cầu 1-1 (Ping Pong)": sum(1 for i in range(len(run_lengths) - 2) if run_lengths[i:i+3] == [1, 1, 1]),
        "Cầu 2-2 (Nhịp đôi)": sum(1 for i in range(len(run_lengths) - 1) if run_lengths[i:i+2] == [2, 2]),
        "Cầu 3-1": sum(1 for i in range(len(run_lengths) - 1) if run_lengths[i:i+2] == [3, 1]),
        "Cầu 1-3": sum(1 for i in range(len(run_lengths) - 1) if run_lengths[i:i+2] == [1, 3]),
        "Cầu 1-2": sum(1 for i in range(len(run_lengths) - 1) if run_lengths[i:i+2] == [1, 2]),
        "Cầu 2-1": sum(1 for i in range(len(run_lengths) - 1) if run_lengths[i:i+2] == [2, 1]),
        "Cầu 2-1-2-1 (Lặp lệch)": sum(1 for i in range(len(run_lengths) - 3) if run_lengths[i:i+4] == [2, 1, 2, 1]),
        "Cầu Tăng Dần (1-2-3)": sum(1 for i in range(len(run_lengths) - 2) if run_lengths[i:i+3] == [1, 2, 3]),
        "Cầu Giảm Dần (3-2-1)": sum(1 for i in range(len(run_lengths) - 2) if run_lengths[i:i+3] == [3, 2, 1]),
        "Cầu Bẻ Giả (Fake Break)": sum(1 for i in range(len(run_lengths) - 2) if run_lengths[i] >= 3 and run_lengths[i+1] == 1 and run_lengths[i+2] >= 3),
        "Cầu 2-4 / 4-2": sum(1 for i in range(len(run_lengths) - 1) if run_lengths[i:i+2] in ([2, 4], [4, 2])),
        "Cầu 3-4 / 4-3": sum(1 for i in range(len(run_lengths) - 1) if run_lengths[i:i+2] in ([3, 4], [4, 3])),
        "Cầu 2-5-2-4": sum(1 for i in range(len(run_lengths) - 3) if run_lengths[i:i+4] == [2, 5, 2, 4]),
        "Cầu Double Bệt (Bệt Đối Xứng)": sum(1 for i in range(len(run_lengths) - 1) if run_lengths[i] >= 3 and run_lengths[i+1] >= 3)
    }

    bet_t_freq = sum(1 for r in runs[:-1] if r[0] == "T" and r[1] >= 3)
    bet_x_freq = sum(1 for r in runs[:-1] if r[0] == "X" and r[1] >= 3)

    max_streak_t = max([r[1] for r in runs[:-1] if r[0] == "T"] + [0])
    max_streak_x = max([r[1] for r in runs[:-1] if r[0] == "X"] + [0])

    current_val = sequence[-1]
    streak_len = runs[-1][1]

    candidates = []

    # Evaluate Standard Patterns — Change 5: multiply raw_score by pattern_weight
    for p_name, base_count in hist_freq.items():
        multiplier, exp, detail = _score_pattern_tail_match(p_name, run_lengths, current_val)
        if multiplier > 0:
            raw_score = (base_count + 1) * multiplier
            weight = pattern_weights.get(p_name, 1.0)
            raw_score *= weight
            candidates.append({"name": p_name, "raw_score": raw_score, "expected": exp, "detail": detail})

    # Evaluate Bệt
    if current_val == "T" and streak_len >= 2:
        multiplier = 10.0 if streak_len >= 3 else 5.0
        raw_score = (bet_t_freq + 1) * multiplier
        detail = "Đang theo xu hướng T"
        if streak_len > max_streak_t and max_streak_t > 0:
            raw_score *= 0.2
            detail += " (Phá kỷ lục cũ - Báo động gãy)"
        bk = f"Bệt Tài ({streak_len}+)"
        weight = pattern_weights.get(bk, 1.0)
        raw_score *= weight
        candidates.append({"name": bk, "raw_score": raw_score, "expected": "T", "detail": detail})
    elif current_val == "X" and streak_len >= 2:
        multiplier = 10.0 if streak_len >= 3 else 5.0
        raw_score = (bet_x_freq + 1) * multiplier
        detail = "Đang theo xu hướng X"
        if streak_len > max_streak_x and max_streak_x > 0:
            raw_score *= 0.2
            detail += " (Phá kỷ lục cũ - Báo động gãy)"
        bk = f"Bệt Xỉu ({streak_len}+)"
        weight = pattern_weights.get(bk, 1.0)
        raw_score *= weight
        candidates.append({"name": bk, "raw_score": raw_score, "expected": "X", "detail": detail})

    # ── Soft Priority Damping ────────────────────────────────────────
    # When a longer/more-specific pattern is active, reduce overlapping shorter
    # patterns' raw_score by 50% (keeps them visible in UI but prevents them
    # from unfairly dominating the probability blend).
    SOFT_PRIORITY: dict[str, set[str]] = {
        "Cầu 2-2 (Nhịp đôi)":        {"Cầu 1-2", "Cầu 2-1"},
        "Cầu 2-1-2-1 (Lặp lệch)":   {"Cầu 2-1", "Cầu 1-2"},
        "Cầu Tăng Dần (1-2-3)":      {"Cầu 1-2", "Cầu 2-1"},
        "Cầu Giảm Dần (3-2-1)":      {"Cầu 3-1"},
        "Cầu 2-5-2-4":               {"Cầu 2-2 (Nhịp đôi)", "Cầu 2-4 / 4-2"},
    }
    present = {c["name"] for c in candidates}
    for priority_pattern, damped_set in SOFT_PRIORITY.items():
        if priority_pattern in present:
            for c in candidates:
                if c["name"] in damped_set:
                    c["raw_score"] *= 0.5  # reduce, not remove

    # Apply equilibrium constraints (rolling window)
    for c in candidates:
        if c["expected"] == "T" and t_ratio > 0.65:
            c["raw_score"] *= 0.1
            c["detail"] += " (Bị ép giảm do Tỉ lệ T>65%)"
        if c["expected"] == "X" and x_ratio > 0.65:
            c["raw_score"] *= 0.1
            c["detail"] += " (Bị ép giảm do Tỉ lệ X>65%)"

    if not candidates:
        return []

    total_score = sum(c["raw_score"] for c in candidates)
    if total_score == 0:
        return []

    K_NOISE = total_score * 0.8
    normalized_total = total_score + K_NOISE

    for c in candidates:
        base_noise_share = K_NOISE / len(candidates)
        c["prob"] = round((c["raw_score"] + base_noise_share) / normalized_total, 3)

    candidates.sort(key=lambda x: x["prob"], reverse=True)
    top_3 = candidates[:3]

    results = []
    for c in top_3:
        percentage = min(99, int(c["prob"] * 100))
        if percentage > 0:
            results.append({
                "name": c["name"],
                "detail": c["detail"],
                "prob": c["prob"],
                "percentage_str": f"{percentage}%",
                "expected": c["expected"],
            })

    return results


# ─── Bias Detection ──────────────────────────────────────────────────────────

def detect_bias(history: list[dict]) -> dict:
    """Only for dice entries: flag any die face > 25% of appearances."""
    dice_entries = [h for h in history if h.get("source") == "dice" and h.get("dice")]
    if not dice_entries:
        return {"has_bias": False, "biased_faces": [], "face_pcts": {}}

    face_count: dict[int, int] = defaultdict(int)
    total_dice = 0
    for entry in dice_entries:
        for d in entry["dice"]:
            face_count[d] += 1
            total_dice += 1

    face_pcts = {str(face): round(count / total_dice * 100, 1) for face, count in sorted(face_count.items())}
    biased = [face for face, count in face_count.items() if count / total_dice > 0.25]

    return {
        "has_bias": len(biased) > 0,
        "biased_faces": sorted(biased),
        "face_pcts": face_pcts,
    }


# ─── Bias → Suggestion Influence ────────────────────────────────────────────

def apply_bias_influence(probs: dict, bias: dict, history: list[dict]) -> str:
    """If bias detected, shift the suggestion based on biased face value."""
    base_suggest = "T" if probs["T"] >= probs["X"] else "X"
    if not bias["has_bias"]:
        return base_suggest

    biased_faces = bias["biased_faces"]
    high = [f for f in biased_faces if f >= 4]
    low = [f for f in biased_faces if f <= 3]

    bias_t_prob = 0.5
    if len(high) > len(low):
        bias_t_prob = 0.85
    elif len(low) > len(high):
        bias_t_prob = 0.15

    probs["T"] = round(probs["T"] * 0.7 + bias_t_prob * 0.3, 4)
    probs["X"] = round(probs["X"] * 0.7 + (1.0 - bias_t_prob) * 0.3, 4)

    return "T" if probs["T"] >= probs["X"] else "X"


# ─── Run-Length Distribution ─────────────────────────────────────────────────

def _run_length_distribution(sequence: list[str]) -> dict:
    if not sequence:
        return {}
    dist: dict[int, int] = defaultdict(int)
    cnt = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            cnt += 1
        else:
            dist[cnt] += 1
            cnt = 1
    dist[cnt] += 1
    return {str(k): v for k, v in sorted(dist.items())}


# ─── Session Weight Tuning ───────────────────────────────────────────────────

def _tune_weights(pattern_weights: dict, pattern_name: str, correct: bool) -> dict:
    """Adjust a single pattern weight based on correctness."""
    w = pattern_weights.get(pattern_name, 1.0)
    if correct:
        w = min(2.0, w * 1.05)
    else:
        w = max(0.3, w * 0.90)
    pattern_weights[pattern_name] = round(w, 5)
    return pattern_weights


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "TX Prediction Engine v2.0 running"}


# ── Session Management ──────────────────────────────────────────────────────

@app.get("/sessions")
def list_sessions():
    """List all available sessions."""
    sessions = []
    for fname in os.listdir(DATA_DIR):
        if fname.startswith("session_") and fname.endswith(".json"):
            path = os.path.join(DATA_DIR, fname)
            try:
                with open(path, "r") as f:
                    s = json.load(f)
                sessions.append({
                    "session_id": s.get("session_id"),
                    "description": s.get("description", ""),
                    "total_entries": len(s.get("history", [])),
                })
            except Exception:
                pass
    return {"sessions": sessions}


@app.post("/sessions/create")
def create_session(data: CreateSessionInput):
    """Create a new (or overwrite) a session."""
    safe = _safe_session_id(data.session_id)
    session = {
        "session_id": data.session_id,
        "description": data.description,
        "history": [],
        "pattern_weights": {},
    }
    save_session(session)
    return {"session_id": data.session_id, "safe_id": safe, "message": "Session created."}


# ── Input Endpoints ─────────────────────────────────────────────────────────

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), session_id: str = "default"):
    """Process a 20×5 grid image and append 100 results to history."""
    content = await file.read()
    try:
        results, debug_log, low_confidence_cells = process_grid_image(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = load_session(session_id)
    history = session["history"]
    pattern_weights = session.get("pattern_weights", {})

    for r in results:
        sequence = [h["result"] for h in history]
        prediction = _make_prediction_snapshot(history, sequence, pattern_weights)
        history.append(_make_entry(r, "image", None, prediction))

    session["history"] = history
    save_session(session)

    low_count = len(low_confidence_cells)
    return {
        "count": len(results),
        "results": results,
        "debug_log": debug_log,
        "low_confidence_cells": low_confidence_cells,
        "low_confidence_count": low_count,
        "warning": f"{low_count} ô có confidence thấp, kết quả có thể sai." if low_count > 0 else None,
        "message": f"Đã thêm {len(results)} kết quả từ ảnh.",
    }


@app.post("/append-manual")
def append_manual(data: ManualInput):
    """Append a single manual result."""
    if data.result not in ["T", "X"]:
        raise HTTPException(status_code=400, detail="Result must be T or X")

    session = load_session(data.session_id)
    history = session["history"]
    pattern_weights = session.get("pattern_weights", {})
    sequence = [h["result"] for h in history]

    prediction = _make_prediction_snapshot(history, sequence, pattern_weights)
    history.append(_make_entry(data.result, "manual", None, prediction))
    session["history"] = history
    save_session(session)

    return {
        "result": data.result,
        "total_entries": len(history),
        "session_id": data.session_id,
    }


@app.post("/append-dice")
def append_dice(dice_input: DiceInput):
    """Validate dice values and append result to history."""
    for d in [dice_input.d1, dice_input.d2, dice_input.d3]:
        if not (1 <= d <= 6):
            raise HTTPException(status_code=400, detail="Mỗi xúc xắc phải từ 1 đến 6.")

    result = dice_to_result(dice_input.d1, dice_input.d2, dice_input.d3)
    session = load_session(dice_input.session_id)
    history = session["history"]
    pattern_weights = session.get("pattern_weights", {})
    sequence = [h["result"] for h in history]

    prediction = _make_prediction_snapshot(history, sequence, pattern_weights)
    history.append(_make_entry(result, "dice", [dice_input.d1, dice_input.d2, dice_input.d3], prediction))
    session["history"] = history
    save_session(session)

    return {
        "dice": [dice_input.d1, dice_input.d2, dice_input.d3],
        "sum": dice_input.d1 + dice_input.d2 + dice_input.d3,
        "result": result,
        "total_entries": len(history),
        "session_id": dice_input.session_id,
    }


# ── Predict ─────────────────────────────────────────────────────────────────

@app.get("/predict")
def predict(session_id: str = "default"):
    """Return probabilities, suggestion, pattern, blend weights, and bias info."""
    session = load_session(session_id)
    history = session["history"]
    pattern_weights = session.get("pattern_weights", {})
    sequence = [h["result"] for h in history]

    probs = advanced_predict(sequence)
    top_patterns = detect_top_patterns(sequence, pattern_weights)
    bias = detect_bias(history)
    blend_w = compute_blend_weights(sequence, top_patterns)

    if top_patterns:
        # Rank-weighted blend: #1 pattern gets 60% say, #2 gets 25%, #3 gets 15%.
        _RANK_W = [0.60, 0.25, 0.15]
        t_mass = sum(_RANK_W[i] for i, p in enumerate(top_patterns[:3]) if p["expected"] == "T")
        x_mass = sum(_RANK_W[i] for i, p in enumerate(top_patterns[:3]) if p["expected"] == "X")
        _total = t_mass + x_mass
        if _total > 0:
            t_mass /= _total
            x_mass /= _total
        pw = blend_w["pattern"]
        mw = blend_w["markov"]
        probs["T"] = round(probs["T"] * mw + t_mass * pw, 4)
        probs["X"] = round(probs["X"] * mw + x_mass * pw, 4)

    suggest = apply_bias_influence(probs, bias, history)
    accuracy = compute_accuracy(history)

    recent_manual = [
        {
            "source": h.get("source", "unknown"),
            "dice": h.get("dice"),
            "sum": sum(h.get("dice", [])) if h.get("dice") else None,
            "result": h["result"],
        }
        for h in history if h.get("source") in ["dice", "manual"]
    ]

    return {
        "session_id": session_id,
        "total_entries": len(history),
        "sequence_tail": sequence[-100:],
        "recent_manual": recent_manual,
        "probabilities": probs,
        "suggest": suggest,
        "blend_weights": blend_w,
        "patterns": top_patterns,
        "bias": bias,
        "accuracy": accuracy,
    }


# ── Feedback Loop ────────────────────────────────────────────────────────────

@app.post("/confirm")
def confirm(data: ConfirmInput):
    """Mark the last unconfirmed predicted entry, tune weights, return accuracy."""
    if data.actual not in ["T", "X"]:
        raise HTTPException(status_code=400, detail="actual must be 'T' or 'X'")

    session = load_session(data.session_id)
    history = session["history"]
    pattern_weights = session.get("pattern_weights", {})

    # Find last unconfirmed entry with a non-null prediction
    target_idx = None
    for i in range(len(history) - 1, -1, -1):
        pred = history[i].get("prediction")
        if pred and pred.get("suggest") is not None and not pred.get("confirmed"):
            target_idx = i
            break

    if target_idx is None:
        raise HTTPException(status_code=404, detail="No unconfirmed predicted entry found.")

    entry = history[target_idx]
    pred = entry["prediction"]
    was_correct = (pred["suggest"] == data.actual)

    pred["confirmed"] = True
    pred["was_correct"] = was_correct

    # Tune weights: top pattern in the snapshot
    top_pattern = pred.get("top_pattern")
    if top_pattern and isinstance(top_pattern, dict):
        pname = top_pattern.get("name")
        if pname:
            pattern_weights = _tune_weights(pattern_weights, pname, was_correct)

    session["history"] = history
    session["pattern_weights"] = pattern_weights
    save_session(session)

    accuracy = compute_accuracy(history)
    return {
        "session_id": data.session_id,
        "entry_index": target_idx,
        "suggested": pred["suggest"],
        "actual": data.actual,
        "was_correct": was_correct,
        "accuracy": accuracy,
    }


# ── Stats ────────────────────────────────────────────────────────────────────

@app.get("/stats")
def stats(session_id: str = "default"):
    """Return session statistics."""
    session = load_session(session_id)
    history = session["history"]
    sequence = [h["result"] for h in history]
    pattern_weights = session.get("pattern_weights", {})

    t_count = sequence.count("T")
    x_count = sequence.count("X")
    total = len(sequence)

    # Current streak
    current_streak = 0
    if sequence:
        v = sequence[-1]
        for c in reversed(sequence):
            if c == v:
                current_streak += 1
            else:
                break

    run_dist = _run_length_distribution(sequence)
    accuracy = compute_accuracy(history)

    return {
        "session_id": session_id,
        "description": session.get("description", ""),
        "total_entries": total,
        "t_count": t_count,
        "x_count": x_count,
        "t_ratio": round(t_count / total, 4) if total else None,
        "x_ratio": round(x_count / total, 4) if total else None,
        "current_streak": current_streak,
        "current_streak_value": sequence[-1] if sequence else None,
        "run_length_distribution": run_dist,
        "pattern_weights": pattern_weights,
        "accuracy": accuracy,
    }


# ── Reset ────────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(session_id: str = "default"):
    """Clear all history for a session."""
    session = load_session(session_id)
    session["history"] = []
    session["pattern_weights"] = {}
    save_session(session)
    return {"message": "Lịch sử đã được xóa.", "session_id": session_id, "total_entries": 0}
