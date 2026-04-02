import json
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

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(title="TX Prediction Engine", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_SESSION = "default"

# ─── Cầu còn lại sau khi lọc ─────────────────────────────────────────────────
# GIỮ (4 cầu cốt lõi):
#   Cầu 1-1 (Ping Pong), Cầu 2-2 (Nhịp đôi), Bệt T/X, Cầu Bẻ Giả
# CÓ ĐIỀU KIỆN - chỉ dùng khi xuất hiện đủ MIN_FREQ lần:
#   Cầu 3-1, Cầu 1-3, Cầu Tăng Dần (1-2-3), Cầu Giảm Dần (3-2-1), Cầu 2-1-2-1
# ĐÃ BỎ (5 cầu nhiễu):
#   Cầu 1-2, Cầu 2-1       -> quá ngắn, fire liên tục, không đặc hiệu
#   Cầu 2-4/4-2             -> quá hiếm, không đủ mẫu thống kê
#   Cầu 3-4/4-3             -> quá hiếm, không đủ mẫu thống kê
#   Cầu 2-5-2-4             -> cực hiếm, multiplier 12.0 gây lệch nặng
#   Cầu Double Bệt          -> logic trùng hoàn toàn với Bệt + Fake Break

MIN_FREQ: dict[str, int] = {
    "Cầu 3-1":                5,
    "Cầu 1-3":                5,
    "Cầu Tăng Dần (1-2-3)":  3,
    "Cầu Giảm Dần (3-2-1)":  3,
    "Cầu 2-1-2-1 (Lặp lệch)": 2,
}

# ─── Data Models ─────────────────────────────────────────────────────────────

class DiceInput(BaseModel):
    d1: int
    d2: int
    d3: int
    session_id: str = DEFAULT_SESSION

class ManualInput(BaseModel):
    result: str
    session_id: str = DEFAULT_SESSION

class ConfirmInput(BaseModel):
    actual: str
    session_id: str = DEFAULT_SESSION

class CreateSessionInput(BaseModel):
    session_id: str
    description: str = ""

# ─── Session I/O ─────────────────────────────────────────────────────────────

def _safe_id(session_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)[:64]

def _session_path(session_id: str) -> str:
    return os.path.join(DATA_DIR, f"session_{_safe_id(session_id)}.json")

def load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return {"session_id": session_id, "description": "", "history": [], "pattern_weights": {}}
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"session_id": session_id, "description": "", "history": [], "pattern_weights": {}}

def save_session(session: dict):
    with open(_session_path(session["session_id"]), "w") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)

# ─── Image Processing ────────────────────────────────────────────────────────

def process_grid_image(file_bytes: bytes) -> tuple[list[str], list[dict], list[dict]]:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra định dạng file.")

    _, thresh_bright = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(img, 50, 150)
    mask = cv2.bitwise_or(thresh_bright, edges)
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) > 0:
        img = img[
            max(0, int(np.min(y_coords)) - 5): min(img.shape[0], int(np.max(y_coords)) + 5),
            max(0, int(np.min(x_coords)) - 5): min(img.shape[1], int(np.max(x_coords)) + 5),
        ]

    img_resized = cv2.resize(img, (1000, 250), interpolation=cv2.INTER_AREA)
    cells_data = []
    for col in range(20):
        for row in (range(5) if col % 2 == 0 else range(4, -1, -1)):
            cell = img_resized[row*50+10:row*50+40, col*50+10:col*50+40]
            dark_mean = float(np.mean(np.sort(cell.flatten())[:100]))
            cells_data.append({"col": col, "row": row, "dark_mean": dark_mean})

    all_dm  = [c["dark_mean"] for c in cells_data]
    min_p5  = float(np.percentile(all_dm, 5))
    max_p95 = float(np.percentile(all_dm, 95))
    threshold = (min_p5 + max_p95) / 2
    margin    = (max_p95 - min_p5) * 0.15

    results, debug_log, low_conf = [], [], []
    for c in cells_data:
        val = "T" if c["dark_mean"] < threshold else "X"
        low = abs(c["dark_mean"] - threshold) < margin
        results.append(val)
        entry = {
            "time_idx": len(results) - 1,
            "col": c["col"], "row": c["row"],
            "dark_mean": round(c["dark_mean"], 1),
            "result": val,
            "confidence": "low" if low else "ok",
        }
        debug_log.append(entry)
        if low:
            low_conf.append(entry)

    return results, debug_log, low_conf

# ─── Dice ────────────────────────────────────────────────────────────────────

def dice_to_result(d1: int, d2: int, d3: int) -> str:
    return "T" if (d1 + d2 + d3) >= 11 else "X"

# ─── Accuracy ────────────────────────────────────────────────────────────────

def compute_accuracy(history: list[dict]) -> dict:
    confirmed = [
        h for h in history
        if h.get("prediction", {}).get("confirmed")
        and h["prediction"].get("suggest") is not None
    ]
    def _acc(entries):
        if not entries:
            return None
        return round(sum(1 for e in entries if e["prediction"].get("was_correct")) / len(entries) * 100, 2)
    return {
        "total_confirmed":   len(confirmed),
        "accuracy_last_20":  _acc(confirmed[-20:]),
        "accuracy_last_50":  _acc(confirmed[-50:]),
        "accuracy_last_100": _acc(confirmed[-100:]),
        "accuracy_all":      _acc(confirmed),
    }

# ─── Dynamic Blend Weights ────────────────────────────────────────────────────

def compute_blend_weights(sequence: list[str], top_patterns: list[dict]) -> dict:
    n = len(sequence)
    base = 0.20 if n < 20 else 0.40 if n < 50 else 0.55 if n < 100 else 0.65
    if not top_patterns:
        base -= 0.15
    else:
        tp = top_patterns[0]["prob"]
        base += 0.15 if tp >= 0.70 else 0.08 if tp >= 0.60 else 0.0
    pw = max(0.20, min(0.80, base))
    return {"markov": round(1.0 - pw, 4), "pattern": round(pw, 4)}

# ─── Run-Length Predictor ────────────────────────────────────────────────────

def advanced_predict(sequence: list[str]) -> dict:
    if len(sequence) < 3:
        return {"T": 0.5, "X": 0.5}

    cur_val   = sequence[-1]
    other_val = "X" if cur_val == "T" else "T"

    streak_len = 1
    for v in reversed(sequence[:-1]):
        if v == cur_val: streak_len += 1
        else: break

    hist_runs, cur, cnt = [], sequence[0], 1
    for v in sequence[1:]:
        if v == cur: cnt += 1
        else:
            if cur == cur_val: hist_runs.append(cnt)
            cur, cnt = v, 1

    reached   = [r for r in hist_runs if r >= streak_len]
    continued = sum(1 for r in reached if r > streak_len)
    broken    = sum(1 for r in reached if r == streak_len)
    p_cont    = (continued + 1.0) / (continued + broken + 2.0)
    p_broke   = (broken    + 1.0) / (continued + broken + 2.0)

    recent      = sequence[-20:] if len(sequence) >= 20 else sequence
    final_cont  = p_cont  * 0.7 + recent.count(cur_val)   / len(recent) * 0.3
    final_broke = p_broke * 0.7 + recent.count(other_val) / len(recent) * 0.3
    total = final_cont + final_broke

    return {cur_val: round(final_cont/total, 4), other_val: round(final_broke/total, 4)}

# ─── Pattern Engine ───────────────────────────────────────────────────────────

def _score_pattern_tail(name: str, r: list[int], cur: str) -> tuple[float, str, str]:
    sw   = "X" if cur == "T" else "T"
    cont = cur
    if not r:
        return 0.0, cont, ""

    if name == "Cầu 1-1 (Ping Pong)":
        if len(r) >= 6 and all(x == 1 for x in r[-6:]): return 6.0,  sw,   "Ping pong quá dài — chuẩn bị gãy"
        if len(r) >= 3 and r[-3]==1 and r[-2]==1 and r[-1]==1:  return 10.0, sw,   "Đang xen kẽ T/X liên tục"
        if len(r) >= 2 and r[-2]==1 and r[-1]==1:               return 3.0,  sw,   "Dấu hiệu cầu 1-1"
        return 0.0, sw, ""

    if name == "Cầu 2-2 (Nhịp đôi)":
        if len(r) >= 2 and r[-2]==2 and r[-1]==2:               return 8.0,  sw,   "Nhịp đôi đều đặn"
        if len(r) >= 3 and r[-3]==2 and r[-2]==2 and r[-1]==1:  return 10.0, cont, "Chờ hoàn thành nhịp 2"
        if len(r) >= 2 and r[-2]==2 and r[-1]==1:               return 7.0,  cont, "Chờ hoàn thành nhịp 2"
        return 0.0, cont, ""

    if name == "Cầu 2-1-2-1 (Lặp lệch)":
        # Đã hoàn chỉnh — tín hiệu mạnh
        if len(r) >= 4 and r[-4]==2 and r[-3]==1 and r[-2]==2 and r[-1]==1: return 9.0,  cont, "Vừa nhịp 1, chờ 2"
        if len(r) >= 3 and r[-3]==2 and r[-2]==1 and r[-1]==2:              return 10.0, sw,   "Vừa nhịp 2, chờ 1"
        # Đang hình thành — chỉ fire vì history đã có pattern này (MIN_FREQ=2 đã lọc)
        if len(r) >= 2 and r[-2]==2 and r[-1]==1: return 5.0, cont, "Nhịp 2-1 — chờ lặp lại"
        if len(r) >= 2 and r[-2]==1 and r[-1]==2: return 5.0, sw,   "Nhịp 1-2 — chờ lặp lại"
        return 0.0, cont, ""

    if name == "Cầu Tăng Dần (1-2-3)":
        if len(r) >= 2 and r[-2]==1 and r[-1]==2:               return 6.0,  sw,   "Nhịp 1-2 xong, xả bẻ sang 3"
        if len(r) >= 3 and r[-3]==1 and r[-2]==2 and r[-1]==1:  return 8.0,  cont, "Đang lên nhịp 3"
        if len(r) >= 3 and r[-3]==1 and r[-2]==2 and r[-1]==2:  return 9.0,  cont, "Đang lên nhịp 3"
        if len(r) >= 3 and r[-3]==1 and r[-2]==2 and r[-1]==3:  return 8.0,  sw,   "Hoàn thành 1-2-3, xả cầu"
        return 0.0, cont, ""

    if name == "Cầu Giảm Dần (3-2-1)":
        if len(r) >= 2 and r[-2]==3 and r[-1]==2:               return 8.0, sw,   "Nhịp 3-2 xong, bắt nhịp 1"
        if len(r) >= 3 and r[-3]==3 and r[-2]==2 and r[-1]==1:  return 8.0, sw,   "Hoàn thành 3-2-1, xả cầu"
        return 0.0, cont, ""

    if name == "Cầu Bẻ Giả (Fake Break)":
        if len(r) >= 3 and r[-3]>=3 and r[-2]==1 and r[-1]==1:  return 8.0,  cont, "Dấu hiệu bẻ giả, trở lại Bệt"
        if len(r) >= 3 and r[-3]>=3 and r[-2]==1 and r[-1]>=2:  return 10.0, cont, "Bẻ giả thành công, tiếp tục Bệt"
        return 0.0, cont, ""

    if name == "Cầu 3-1":
        if len(r) >= 2 and r[-2]==3 and r[-1]==1:               return 8.0, sw,   "Xong nhịp 3-1"
        if len(r) >= 3 and r[-3]==3 and r[-2]==1 and r[-1]<=2:  return 8.0, cont, "Chờ nhịp 3"
        return 0.0, cont, ""

    if name == "Cầu 1-3":
        if len(r) >= 2 and r[-2]==1 and r[-1]==3:  return 8.0, sw,   "Xong nhịp 1-3"
        if len(r) >= 2 and r[-2]==1 and r[-1]<=2:  return 6.0, cont, "Chờ nhịp 3"
        return 0.0, cont, ""

    return 0.0, cont, ""


def detect_top_patterns(sequence: list[str], pattern_weights: Optional[dict] = None) -> list[dict]:
    if len(sequence) < 5:
        return []
    if pattern_weights is None:
        pattern_weights = {}

    window  = sequence[-30:]
    t_ratio = window.count("T") / len(window)
    x_ratio = window.count("X") / len(window)

    runs: list[tuple[str, int]] = []
    cur, cnt = sequence[0], 1
    for v in sequence[1:]:
        if v == cur: cnt += 1
        else:
            runs.append((cur, cnt))
            cur, cnt = v, 1
    runs.append((cur, cnt))
    rl = [r[1] for r in runs]

    hist_freq = {
        "Cầu 1-1 (Ping Pong)":     sum(1 for i in range(len(rl)-2) if rl[i:i+3]==[1,1,1]),
        "Cầu 2-2 (Nhịp đôi)":      sum(1 for i in range(len(rl)-1) if rl[i:i+2]==[2,2]),
        "Cầu 3-1":                  sum(1 for i in range(len(rl)-1) if rl[i:i+2]==[3,1]),
        "Cầu 1-3":                  sum(1 for i in range(len(rl)-1) if rl[i:i+2]==[1,3]),
        "Cầu 2-1-2-1 (Lặp lệch)":  sum(1 for i in range(len(rl)-3) if rl[i:i+4]==[2,1,2,1]),
        "Cầu Tăng Dần (1-2-3)":    sum(1 for i in range(len(rl)-2) if rl[i:i+3]==[1,2,3]),
        "Cầu Giảm Dần (3-2-1)":    sum(1 for i in range(len(rl)-2) if rl[i:i+3]==[3,2,1]),
        "Cầu Bẻ Giả (Fake Break)": sum(1 for i in range(len(rl)-2) if rl[i]>=3 and rl[i+1]==1 and rl[i+2]>=3),
    }

    bet_t = sum(1 for r in runs[:-1] if r[0]=="T" and r[1]>=3)
    bet_x = sum(1 for r in runs[:-1] if r[0]=="X" and r[1]>=3)
    max_t = max([r[1] for r in runs[:-1] if r[0]=="T"] + [0])
    max_x = max([r[1] for r in runs[:-1] if r[0]=="X"] + [0])

    cur_val    = sequence[-1]
    streak_len = runs[-1][1]
    candidates = []

    for p_name, freq in hist_freq.items():
        if freq < MIN_FREQ.get(p_name, 1):
            continue
        mult, expected, detail = _score_pattern_tail(p_name, rl, cur_val)
        if mult <= 0:
            continue
        raw = (freq + 1) * mult * pattern_weights.get(p_name, 1.0)
        candidates.append({"name": p_name, "raw_score": raw, "expected": expected,
                            "detail": detail, "freq": freq})

    if cur_val == "T" and streak_len >= 2:
        mult   = 10.0 if streak_len >= 3 else 5.0
        raw    = (bet_t + 1) * mult
        detail = "Đang theo xu hướng T"
        if streak_len > max_t > 0:
            raw *= 0.2; detail += " (Phá kỷ lục — cảnh báo gãy)"
        bk   = f"Bệt Tài ({streak_len}+)"
        raw *= pattern_weights.get(bk, 1.0)
        candidates.append({"name": bk, "raw_score": raw, "expected": "T", "detail": detail, "freq": bet_t})

    elif cur_val == "X" and streak_len >= 2:
        mult   = 10.0 if streak_len >= 3 else 5.0
        raw    = (bet_x + 1) * mult
        detail = "Đang theo xu hướng X"
        if streak_len > max_x > 0:
            raw *= 0.2; detail += " (Phá kỷ lục — cảnh báo gãy)"
        bk   = f"Bệt Xỉu ({streak_len}+)"
        raw *= pattern_weights.get(bk, 1.0)
        candidates.append({"name": bk, "raw_score": raw, "expected": "X", "detail": detail, "freq": bet_x})

    SOFT_PRIORITY = {
        "Cầu 2-2 (Nhịp đôi)":   {"Cầu 2-1-2-1 (Lặp lệch)"},
        "Cầu Tăng Dần (1-2-3)": {"Cầu 3-1"},
        "Cầu Giảm Dần (3-2-1)": {"Cầu 3-1"},
    }
    present = {c["name"] for c in candidates}
    for priority, damped in SOFT_PRIORITY.items():
        if priority in present:
            for c in candidates:
                if c["name"] in damped:
                    c["raw_score"] *= 0.5

    for c in candidates:
        if c["expected"] == "T" and t_ratio > 0.65:
            c["raw_score"] *= 0.1; c["detail"] += " (Nén: T >65% trong 30 ván gần)"
        if c["expected"] == "X" and x_ratio > 0.65:
            c["raw_score"] *= 0.1; c["detail"] += " (Nén: X >65% trong 30 ván gần)"

    if not candidates:
        return []

    # Softmax thay K_NOISE — giữ thông tin confidence thực sự
    TEMPERATURE = 2.0
    scores = np.array([c["raw_score"] for c in candidates], dtype=float)
    scores = scores / TEMPERATURE
    scores -= scores.max()
    exp_s  = np.exp(scores)
    probs  = exp_s / exp_s.sum()

    for i, c in enumerate(candidates):
        c["prob"] = round(float(probs[i]), 4)

    candidates.sort(key=lambda x: x["prob"], reverse=True)

    return [
        {
            "name":           c["name"],
            "detail":         c["detail"],
            "prob":           c["prob"],
            "percentage_str": f"{min(99, int(c['prob']*100))}%",
            "expected":       c["expected"],
            "freq":           c.get("freq", 0),
        }
        for c in candidates[:3] if int(c["prob"]*100) > 0
    ]

# ─── Confidence Gate ─────────────────────────────────────────────────────────

def compute_signal(probs: dict, top_patterns: list[dict], sequence: list[str]) -> tuple[Optional[str], str]:
    """
    3 tín hiệu độc lập bỏ phiếu.
    Cần >= 2/3 đồng thuận mới output T/X.
    Ngược chiều -> suggest=None (Không rõ cầu - bỏ ván này).
    """
    votes: list[str] = []

    # Tín hiệu 1: Markov
    markov_side = "T" if probs["T"] >= probs["X"] else "X"
    if max(probs["T"], probs["X"]) >= 0.55:
        votes.append(markov_side)

    # Tín hiệu 2: Pattern
    if top_patterns and top_patterns[0]["prob"] >= 0.55:
        votes.append(top_patterns[0]["expected"])

    # Tín hiệu 3: Momentum 20 ván gần
    recent = sequence[-20:] if len(sequence) >= 20 else sequence
    if recent:
        tr = recent.count("T") / len(recent)
        xr = recent.count("X") / len(recent)
        if tr >= 0.60:   votes.append("T")
        elif xr >= 0.60: votes.append("X")

    if not votes:
        return None, "none"

    t_v = votes.count("T")
    x_v = votes.count("X")

    if t_v >= 2 and t_v > x_v:
        return "T", ("strong" if t_v == 3 else "moderate")
    if x_v >= 2 and x_v > t_v:
        return "X", ("strong" if x_v == 3 else "moderate")

    return None, "none"

# ─── Bias ────────────────────────────────────────────────────────────────────

def detect_bias(history: list[dict]) -> dict:
    dice_entries = [h for h in history if h.get("source") == "dice" and h.get("dice")]
    if not dice_entries:
        return {"has_bias": False, "biased_faces": [], "face_pcts": {}}
    fc: dict[int, int] = defaultdict(int)
    total = 0
    for e in dice_entries:
        for d in e["dice"]:
            fc[d] += 1; total += 1
    face_pcts = {str(f): round(c/total*100, 1) for f, c in sorted(fc.items())}
    biased = [f for f, c in fc.items() if c/total > 0.25]
    return {"has_bias": bool(biased), "biased_faces": sorted(biased), "face_pcts": face_pcts}

def apply_bias(probs: dict, bias: dict) -> dict:
    if not bias["has_bias"]:
        return probs
    high = [f for f in bias["biased_faces"] if f >= 4]
    low  = [f for f in bias["biased_faces"] if f <= 3]
    bt   = 0.85 if len(high) > len(low) else 0.15 if len(low) > len(high) else 0.5
    return {
        "T": round(probs["T"] * 0.7 + bt * 0.3, 4),
        "X": round(probs["X"] * 0.7 + (1.0 - bt) * 0.3, 4),
    }

# ─── Snapshot ────────────────────────────────────────────────────────────────

def _make_snapshot(history: list[dict], sequence: list[str], pw: dict) -> dict:
    null = {"suggest": None, "prob_t": None, "prob_x": None, "top_pattern": None,
            "confirmed": False, "was_correct": None, "signal_strength": "none"}
    if len(sequence) < 3:
        return null

    probs        = advanced_predict(sequence)
    top_patterns = detect_top_patterns(sequence, pw)
    bias         = detect_bias(history)
    blend        = compute_blend_weights(sequence, top_patterns)
    probs        = apply_bias(probs, bias)

    if top_patterns:
        t_mass = sum(p["prob"] for p in top_patterns if p["expected"] == "T")
        x_mass = sum(p["prob"] for p in top_patterns if p["expected"] == "X")
        probs["T"] = round(probs["T"] * blend["markov"] + t_mass * blend["pattern"], 4)
        probs["X"] = round(probs["X"] * blend["markov"] + x_mass * blend["pattern"], 4)

    suggest, strength = compute_signal(probs, top_patterns, sequence)
    return {
        "suggest":         suggest,
        "prob_t":          probs.get("T"),
        "prob_x":          probs.get("X"),
        "top_pattern":     top_patterns[0] if top_patterns else None,
        "confirmed":       False,
        "was_correct":     None,
        "signal_strength": strength,
    }

def _make_entry(result: str, source: str, dice: Optional[list], snap: dict) -> dict:
    return {
        "ts":         datetime.datetime.utcnow().isoformat() + "Z",
        "result":     result,
        "source":     source,
        "dice":       dice,
        "prediction": snap,
    }

# ─── Weight Tuning ───────────────────────────────────────────────────────────

def _tune(weights: dict, name: str, correct: bool) -> dict:
    w = weights.get(name, 1.0)
    w = min(2.0, w * 1.05) if correct else max(0.3, w * 0.90)
    weights[name] = round(w, 5)
    return weights

def _run_dist(sequence: list[str]) -> dict:
    if not sequence:
        return {}
    dist: dict[int, int] = defaultdict(int)
    cnt = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]: cnt += 1
        else: dist[cnt] += 1; cnt = 1
    dist[cnt] += 1
    return {str(k): v for k, v in sorted(dist.items())}

# ─── API ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "TX Prediction Engine v2.1 running", "docs": "/docs"}

@app.get("/sessions")
def list_sessions():
    out = []
    for fname in os.listdir(DATA_DIR):
        if fname.startswith("session_") and fname.endswith(".json"):
            try:
                with open(os.path.join(DATA_DIR, fname)) as f:
                    s = json.load(f)
                out.append({"session_id": s.get("session_id"), "description": s.get("description", ""),
                             "total_entries": len(s.get("history", []))})
            except Exception:
                pass
    return {"sessions": out}

@app.post("/sessions/create")
def create_session(data: CreateSessionInput):
    sess = {"session_id": data.session_id, "description": data.description,
            "history": [], "pattern_weights": {}}
    save_session(sess)
    return {"session_id": data.session_id, "message": "Session created."}

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), session_id: str = DEFAULT_SESSION):
    content = await file.read()
    try:
        results, debug_log, low_conf = process_grid_image(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sess     = load_session(session_id)
    history  = sess["history"]
    pw       = sess.get("pattern_weights", {})
    sequence = [h["result"] for h in history]

    # FIX data leakage: snapshot 1 lần TRƯỚC khi append toàn bộ batch
    snap = _make_snapshot(history, sequence, pw)
    for r in results:
        history.append(_make_entry(r, "image", None, snap))

    sess["history"] = history
    save_session(sess)

    lc = len(low_conf)
    return {
        "session_id":           session_id,
        "count":                len(results),
        "results":              results,
        "low_confidence_cells": low_conf,
        "low_confidence_count": lc,
        "warning":              f"{lc} ô nhận diện không chắc — nên kiểm tra thủ công." if lc else None,
        "debug_log":            debug_log,
    }

@app.post("/append-manual")
def append_manual(data: ManualInput):
    if data.result not in ("T", "X"):
        raise HTTPException(status_code=400, detail="result phải là T hoặc X")
    sess     = load_session(data.session_id)
    history  = sess["history"]
    sequence = [h["result"] for h in history]
    snap     = _make_snapshot(history, sequence, sess.get("pattern_weights", {}))
    history.append(_make_entry(data.result, "manual", None, snap))
    sess["history"] = history
    save_session(sess)
    return {"session_id": data.session_id, "result": data.result,
            "total_entries": len(history), "prediction_made": snap["suggest"],
            "signal_strength": snap["signal_strength"]}

@app.post("/append-dice")
def append_dice(data: DiceInput):
    for d in [data.d1, data.d2, data.d3]:
        if not (1 <= d <= 6):
            raise HTTPException(status_code=400, detail="Mỗi xúc xắc phải từ 1 đến 6.")
    result   = dice_to_result(data.d1, data.d2, data.d3)
    sess     = load_session(data.session_id)
    history  = sess["history"]
    sequence = [h["result"] for h in history]
    snap     = _make_snapshot(history, sequence, sess.get("pattern_weights", {}))
    history.append(_make_entry(result, "dice", [data.d1, data.d2, data.d3], snap))
    sess["history"] = history
    save_session(sess)
    return {"session_id": data.session_id, "dice": [data.d1, data.d2, data.d3],
            "sum": data.d1+data.d2+data.d3, "result": result,
            "total_entries": len(history), "prediction_made": snap["suggest"],
            "signal_strength": snap["signal_strength"]}

@app.get("/predict")
def predict(session_id: str = DEFAULT_SESSION):
    sess     = load_session(session_id)
    history  = sess["history"]
    pw       = sess.get("pattern_weights", {})
    sequence = [h["result"] for h in history]

    probs        = advanced_predict(sequence)
    top_patterns = detect_top_patterns(sequence, pw)
    bias         = detect_bias(history)
    blend        = compute_blend_weights(sequence, top_patterns)
    probs        = apply_bias(probs, bias)

    if top_patterns:
        t_mass = sum(p["prob"] for p in top_patterns if p["expected"] == "T")
        x_mass = sum(p["prob"] for p in top_patterns if p["expected"] == "X")
        probs["T"] = round(probs["T"] * blend["markov"] + t_mass * blend["pattern"], 4)
        probs["X"] = round(probs["X"] * blend["markov"] + x_mass * blend["pattern"], 4)

    suggest, strength = compute_signal(probs, top_patterns, sequence)

    return {
        "session_id":      session_id,
        "total_entries":   len(history),
        "sequence_tail":   sequence[-100:],
        "probabilities":   probs,
        "suggest":         suggest,
        "signal_strength": strength,
        "blend_weights":   blend,
        "patterns":        top_patterns,
        "bias":            bias,
        "accuracy":        compute_accuracy(history),
    }

@app.post("/confirm")
def confirm(data: ConfirmInput):
    if data.actual not in ("T", "X"):
        raise HTTPException(status_code=400, detail="actual phải là T hoặc X")
    sess    = load_session(data.session_id)
    history = sess["history"]
    pw      = sess.get("pattern_weights", {})

    target = None
    for i in range(len(history)-1, -1, -1):
        pred = history[i].get("prediction", {})
        if pred.get("suggest") is not None and not pred.get("confirmed"):
            target = i
            break

    if target is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy entry chưa confirm.")

    pred             = history[target]["prediction"]
    was_correct      = pred["suggest"] == data.actual
    pred["confirmed"]   = True
    pred["was_correct"] = was_correct

    tp = pred.get("top_pattern")
    if tp and isinstance(tp, dict) and tp.get("name"):
        pw = _tune(pw, tp["name"], was_correct)

    sess["history"]         = history
    sess["pattern_weights"] = pw
    save_session(sess)

    return {
        "session_id":  data.session_id,
        "entry_index": target,
        "suggested":   pred["suggest"],
        "actual":      data.actual,
        "was_correct": was_correct,
        "accuracy":    compute_accuracy(history),
    }

@app.get("/stats")
def stats(session_id: str = DEFAULT_SESSION):
    sess     = load_session(session_id)
    history  = sess["history"]
    sequence = [h["result"] for h in history]
    total    = len(sequence)

    streak_val = sequence[-1] if sequence else None
    streak_len = 0
    if sequence:
        for v in reversed(sequence):
            if v == streak_val: streak_len += 1
            else: break

    t = sequence.count("T")
    x = sequence.count("X")

    return {
        "session_id":              session_id,
        "description":             sess.get("description", ""),
        "total_entries":           total,
        "t_count":                 t,
        "x_count":                 x,
        "t_ratio":                 round(t/total, 4) if total else None,
        "x_ratio":                 round(x/total, 4) if total else None,
        "current_streak_value":    streak_val,
        "current_streak_length":   streak_len,
        "run_length_distribution": _run_dist(sequence),
        "pattern_weights":         sess.get("pattern_weights", {}),
        "accuracy":                compute_accuracy(history),
    }

@app.post("/reset")
def reset(session_id: str = DEFAULT_SESSION):
    sess = load_session(session_id)
    sess["history"]         = []
    sess["pattern_weights"] = {}
    save_session(sess)
    return {"message": f"Đã xóa session '{session_id}'.", "total_entries": 0}