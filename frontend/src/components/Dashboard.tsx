import type { PredictResponse, Side } from '../api';

interface DashboardProps {
  data: PredictResponse | null;
  loading: boolean;
}

function ProbBar({ label, prob, color }: { label: Side; prob: number; color: 'indigo' | 'rose' }) {
  const pct = Math.round(prob * 100);
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center text-sm">
        <span className={`font-bold ${color === 'indigo' ? 'text-indigo-300' : 'text-rose-300'}`}>
          {label}
        </span>
        <span className="font-mono font-bold text-white text-base">{pct}%</span>
      </div>
      <div className="prob-bar">
        <div
          className={`prob-fill-${label.toLowerCase()}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export default function Dashboard({ data, loading }: DashboardProps) {
  if (loading && !data) {
    return (
      <div className="glass-card p-8 flex items-center justify-center h-48">
        <div className="text-slate-400 text-sm animate-pulse">Đang tải dự đoán…</div>
      </div>
    );
  }

  if (!data || data.total_entries < 2) {
    return (
      <div className="glass-card p-8 flex flex-col items-center justify-center h-48 gap-3">
        <div className="text-4xl">🎲</div>
        <p className="text-slate-400 text-sm text-center">
          Nhập ít nhất 2 kết quả để bắt đầu dự đoán
        </p>
      </div>
    );
  }

  const { probabilities, suggest, patterns, bias, total_entries, recent_manual } = data;

  return (
    <div className="space-y-4">
      {/* Main Prediction Card */}
      <div className="glass-card p-5">
        <p className="section-label mb-4">Dự Đoán Tiếp Theo</p>

        {/* Big suggest badge */}
        <div className="flex items-center gap-5 mb-6">
          <div
            className={`rounded-2xl px-8 py-5 flex flex-col items-center ${
              suggest === 'T' ? 'suggest-t' : 'suggest-x'
            }`}
          >
            <span className="text-5xl font-black font-mono tracking-tight">{suggest}</span>
            <span className="text-xs mt-1 opacity-70">{suggest === 'T' ? 'TÀI' : 'XỈU'}</span>
          </div>
          <div className="flex-1 space-y-4">
            <ProbBar label="T" prob={probabilities.T} color="indigo" />
            <ProbBar label="X" prob={probabilities.X} color="rose" />
          </div>
        </div>

        <div className="text-xs text-slate-500 text-right">
          Tổng bản ghi: <span className="font-mono text-slate-300">{total_entries}</span>
        </div>
      </div>

      {/* Advanced Pattern Analysis */}
      <div className="glass-card p-5">
        <h3 className="section-label mb-3 flex items-center gap-2">
          <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Dự Báo Các Nhịp Cầu (Top 3)
        </h3>
        
        {(!patterns || patterns.length === 0) ? (
          <p className="text-slate-400 text-sm">Chưa có nhịp cầu chuẩn nào được hình thành. Đang tiếp tục theo dõi...</p>
        ) : (
          <div className="space-y-3">
            {patterns.map((p: any, i: number) => {
              const isRisky = p.prob <= 0.56;
              const barColor = isRisky ? 'bg-amber-500' : (p.expected === 'T' ? 'bg-indigo-500' : 'bg-rose-500');
              const badgeStyle = isRisky ? 'bg-amber-500/20 text-amber-300' : (p.expected === 'T' ? 'bg-indigo-500/20 text-indigo-300' : 'bg-rose-500/20 text-rose-300');
              const badgeText = isRisky ? `⚠️ Đánh ${p.expected}` : `Đánh ${p.expected}`;
              
              return (
                <div key={i} className={`flex flex-col gap-1.5 p-3 rounded border ${isRisky ? 'bg-amber-500/10 border-amber-500/20 shadow-[0_0_15px_rgba(245,158,11,0.05)]' : 'bg-white/5 border-white/5'}`}>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-semibold text-white">{p.name} <span className={`text-xs font-normal ${isRisky ? 'text-amber-400' : 'text-slate-400'}`}>({p.percentage_str})</span></span>
                    <span className={`px-2.5 py-0.5 rounded text-xs font-bold whitespace-nowrap transition-colors ${badgeStyle}`}>
                      {badgeText}
                    </span>
                  </div>
                  {/* Progress Bar */}
                  <div className={`w-full rounded-full h-1.5 mt-1 overflow-hidden ${isRisky ? 'bg-amber-950/30' : 'bg-slate-800'}`}>
                    <div 
                      className={`h-1.5 rounded-full ${barColor}`}
                      style={{ width: p.percentage_str }}
                    ></div>
                  </div>
                  <p className={`text-[11px] mt-0.5 ${isRisky ? 'text-amber-400/80' : 'text-slate-400'}`}>{p.detail}</p>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Bias Card */}
      {Object.keys(bias.face_pcts).length > 0 && (
        <div className={`glass-card p-5 ${bias.has_bias ? 'border-amber-500/30' : ''}`}>
          <div className="flex items-center justify-between mb-3">
            <p className="section-label">Phân Tích Bias Xúc Xắc</p>
            {bias.has_bias && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/25 font-medium">
                ⚠ BIAS
              </span>
            )}
          </div>
          <div className="grid grid-cols-6 gap-2">
            {Object.entries(bias.face_pcts).map(([face, pct]) => {
              const isBiased = bias.biased_faces.includes(Number(face));
              return (
                <div
                  key={face}
                  className={`rounded-lg p-2 text-center ${
                    isBiased
                      ? 'bg-amber-500/15 border border-amber-500/30'
                      : 'bg-white/4 border border-white/8'
                  }`}
                >
                  <p className="font-mono font-bold text-base text-white">
                    {['⚀','⚁','⚂','⚃','⚄','⚅'][Number(face) - 1]}
                  </p>
                  <p className="font-mono font-bold text-sm text-white">{face}</p>
                  <p className={`text-xs font-mono mt-0.5 ${isBiased ? 'text-amber-300' : 'text-slate-400'}`}>
                    {pct}%
                  </p>
                </div>
              );
            })}
          </div>
          {bias.has_bias && (
            <p className="text-xs text-amber-300/80 mt-3">
              Mặt {bias.biased_faces.join(', ')} xuất hiện nhiều bất thường → ảnh hưởng gợi ý.
            </p>
          )}
        </div>
      )}

      {/* Recent Manual/Dice Sequence */}
      {recent_manual && recent_manual.length > 0 && (
        <div className="glass-card p-5">
          <div className="flex justify-between items-center mb-3">
            <h3 className="section-label m-0">Lịch Sử Trực Tiếp (Vô hạn)</h3>
            <span className="text-xs font-semibold bg-white/10 px-2.5 py-0.5 rounded-full text-white">{recent_manual.length} kết quả</span>
          </div>
          <div className="flex flex-wrap items-center gap-1.5 max-h-48 overflow-y-auto pr-1">
            {recent_manual.map((entry, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <div className={`flex items-center border rounded overflow-hidden ${entry.result === 'T' ? 'border-indigo-500/40' : 'border-rose-500/40'}`}>
                  <span className={`px-2.5 py-1 text-sm font-bold ${entry.result === 'T' ? 'bg-indigo-500/20 text-indigo-300' : 'bg-rose-500/20 text-rose-300'}`}>
                    {entry.result}
                  </span>
                  {entry.source === 'dice' && (
                    <span className="px-1.5 py-1 text-xs font-mono font-medium text-slate-300 bg-black/30 border-l border-white/5" title="Tổng 3 xúc xắc">
                      {entry.sum}
                    </span>
                  )}
                </div>
                {i < recent_manual.length - 1 && (
                  <div className="w-2 h-[2px] bg-white/20 rounded-full" />
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
