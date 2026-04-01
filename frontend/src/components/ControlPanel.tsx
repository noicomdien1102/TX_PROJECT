import { useRef, useState } from 'react';
import { api, type Side } from '../api';

interface ControlPanelProps {
  onDiceAdded: () => void;
  onManualAdded: () => void;
  onImageUploaded: (results: string[]) => void;
  onReset: () => void;
  onToast: (msg: string, type: 'success' | 'error') => void;
  loading: boolean;
  setLoading: (v: boolean) => void;
}

export default function ControlPanel({
  onDiceAdded,
  onManualAdded,
  onImageUploaded,
  onReset,
  onToast,
  loading,
  setLoading,
}: ControlPanelProps) {
  const [dice, setDice] = useState<[string, string, string]>(['', '', '']);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleDiceChange = (idx: 0 | 1 | 2, val: string) => {
    const next: [string, string, string] = [...dice] as [string, string, string];
    const n = parseInt(val, 10);
    next[idx] = isNaN(n) ? '' : String(Math.max(1, Math.min(6, n)));
    setDice(next);
  };

  const submitDice = async () => {
    const nums = dice.map(Number);
    if (nums.some((n) => n < 1 || n > 6)) {
      onToast('Mỗi xúc xắc phải từ 1 đến 6.', 'error');
      return;
    }
    setLoading(true);
    try {
      const res = await api.appendDice(nums[0], nums[1], nums[2]);
      onToast(`Tổng ${res.sum} → ${res.result} | ${res.total_entries} bản ghi`, 'success');
      setDice(['', '', '']);
      onDiceAdded();
    } catch (e: any) {
      onToast(e.message ?? 'Lỗi không xác định', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    try {
      const res = await api.uploadImage(file);
      console.log('--- DEBUG LOG TỪ ẢNH ---');
      console.table(res.debug_log || []);
      console.log('Gửi thông báo trên cho AI để kiểm tra nhé!');
      onToast(res.message, 'success');
      onImageUploaded(res.results);
    } catch (e: any) {
      onToast(e.message ?? 'Lỗi upload ảnh', 'error');
    } finally {
      setLoading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const handleManualApp = async (result: Side) => {
    setLoading(true);
    try {
      const res = await api.appendManual(result);
      onToast(`Thêm thủ công: ${res.result} | ${res.total_entries} bản ghi`, 'success');
      onManualAdded();
    } catch (e: any) {
      onToast(e.message ?? 'Lỗi thêm thủ công', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if (!confirm('Xóa toàn bộ lịch sử? Hành động không thể hoàn tác.')) return;
    setLoading(true);
    try {
      await api.reset();
      onToast('Đã xóa toàn bộ lịch sử.', 'success');
      onReset();
    } catch (e: any) {
      onToast(e.message ?? 'Lỗi reset', 'error');
    } finally {
      setLoading(false);
    }
  };

  const diceValid = dice.every((d) => d !== '' && Number(d) >= 1 && Number(d) <= 6);

  return (
    <div className="glass-card p-5 space-y-6">
      <div>
        <p className="section-label mb-3">Nhập Xúc Xắc</p>
        <div className="flex items-center gap-3 flex-wrap">
          {([0, 1, 2] as const).map((i) => (
            <div key={i} className="flex flex-col items-center gap-1">
              <span className="text-xs text-slate-400">D{i + 1}</span>
              <input
                id={`dice-input-${i + 1}`}
                type="number"
                min={1}
                max={6}
                value={dice[i]}
                onChange={(e) => handleDiceChange(i, e.target.value)}
                className="dice-input"
                placeholder="—"
                disabled={loading}
              />
            </div>
          ))}
          <div className="flex-1 min-w-[120px]">
            {dice.some((d) => d !== '') && (
              <div className="text-sm text-slate-400 mb-2">
                Tổng:{' '}
                <span className="font-bold text-white font-mono">
                  {dice.filter(Boolean).reduce((a, b) => a + Number(b), 0)}
                </span>
                {diceValid && (
                  <span
                    className={`ml-2 font-bold ${
                      dice.reduce((a, b) => a + Number(b), 0) >= 11
                        ? 'text-indigo-400'
                        : 'text-rose-400'
                    }`}
                  >
                    → {dice.reduce((a, b) => a + Number(b), 0) >= 11 ? 'T' : 'X'}
                  </span>
                )}
              </div>
            )}
            <button
              id="btn-submit-dice"
              className="btn-primary w-full"
              onClick={submitDice}
              disabled={!diceValid || loading}
            >
              {loading ? '...' : 'Thêm Kết Quả'}
            </button>
          </div>
        </div>
      </div>

      <div className="border-t border-white/5 pt-5">
        <div className="flex justify-between items-center mb-3">
          <p className="section-label">Nhập Nhanh Thủ Công</p>
          <span className="text-xs text-slate-500">Click liên tục để điền</span>
        </div>
        <div className="flex gap-3">
          <button
            className="flex-1 bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-300 font-bold py-3 rounded-xl border border-indigo-500/30 transition-all font-mono text-xl"
            onClick={() => handleManualApp('T')}
            disabled={loading}
          >
            + T
          </button>
          <button
            className="flex-1 bg-rose-600/20 hover:bg-rose-600/40 text-rose-300 font-bold py-3 rounded-xl border border-rose-500/30 transition-all font-mono text-xl"
            onClick={() => handleManualApp('X')}
            disabled={loading}
          >
            + X
          </button>
        </div>
      </div>

      <div className="border-t border-white/5 pt-5">
        <p className="section-label mb-3">Upload Lưới Ảnh</p>
        <div className="flex gap-3 flex-wrap items-center">
          <label
            id="btn-upload-image"
            className={`btn-secondary cursor-pointer ${loading ? 'opacity-50 pointer-events-none' : ''}`}
            htmlFor="image-upload"
          >
            📷 Chọn Ảnh (20×5)
          </label>
          <input
            id="image-upload"
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
            disabled={loading}
          />
          <span className="text-xs text-slate-500">PNG, JPG, WebP · Ảnh sẽ resize về 500×500</span>
        </div>
      </div>

      <div className="border-t border-white/5 pt-5 flex justify-end">
        <button
          id="btn-reset"
          className="btn-danger"
          onClick={handleReset}
          disabled={loading}
        >
          🗑 Xóa Lịch Sử
        </button>
      </div>
    </div>
  );
}
