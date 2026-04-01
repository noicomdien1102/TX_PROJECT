import { useEffect, useRef, useState } from 'react';
import type { Side } from '../api';

interface GridProps {
  cells: Side[];
}

export default function Grid({ cells }: GridProps) {
  // Pad to 100 cells
  const padded = [...cells];
  while (padded.length < 100) padded.push(null as unknown as Side);
  const display = padded.slice(0, 100);

  const containerRef = useRef<HTMLDivElement>(null);
  const [linePoints, setLinePoints] = useState<string>('');

  useEffect(() => {
    const updateLines = () => {
      if (!containerRef.current) return;
      const domCells = Array.from(containerRef.current.querySelectorAll('.grid-cell')) as HTMLElement[];
      if (!domCells.length) return;

      const containerRect = containerRef.current.getBoundingClientRect();
      const pts: string[] = [];

      for (let i = 0; i < display.length; i++) {
        if (!display[i]) break; // stop line at first empty cell

        const el = domCells[i];
        if (!el) break;
        const rect = el.getBoundingClientRect();
        // Calculate center of cell relative to the container
        const cx = rect.left - containerRect.left + rect.width / 2;
        const cy = rect.top - containerRect.top + rect.height / 2;
        pts.push(`${cx},${cy}`);
      }
      setLinePoints(pts.join(' '));
    };

    // Use ResizeObserver for perfect responsiveness
    const observer = new ResizeObserver(updateLines);
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }
    
    updateLines();
    
    return () => observer.disconnect();
  }, [display]);

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="section-label mb-1">Visual Grid</p>
          <h2 className="text-base font-semibold text-white">20 × 5 Lưới Kết Quả</h2>
        </div>
        <div className="flex gap-4 text-xs text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-indigo-500 inline-block" />
            T (Tài)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-white inline-block" />
            X (Xỉu)
          </span>
        </div>
      </div>
      
      <div className="relative" ref={containerRef}>
        <svg 
          className="absolute inset-0 pointer-events-none w-full h-full" 
          style={{ zIndex: 10 }}
        >
          {linePoints && (
            <polyline 
              points={linePoints} 
              fill="none" 
              stroke="rgba(255, 255, 255, 0.5)" 
              strokeWidth="2" 
              strokeLinejoin="round" 
              strokeLinecap="round" 
            />
          )}
        </svg>

        <div
          style={{ display: 'grid', gridTemplateColumns: 'repeat(20, 1fr)', gap: '8px' }}
        >
          {display.map((cell, i) => {
            // Reconstruct snake pattern for CSS Grid (which is 1-indexed)
            const col = Math.floor(i / 5);
            const rowInCol = i % 5;
            const row = col % 2 === 0 ? rowInCol : 4 - rowInCol;
            
            return (
              <div
                key={i}
                className={`grid-cell ${
                  cell === 'T' ? 'cell-t' : cell === 'X' ? 'cell-x' : 'cell-empty'
                }`}
                style={{ gridColumn: col + 1, gridRow: row + 1, position: 'relative', zIndex: 20 }}
                title={cell ?? '—'}
              />
            );
          })}
        </div>
      </div>
      
      <p className="text-center text-xs text-slate-500 mt-3">
        {cells.length} / 100 ô đã có dữ liệu
      </p>
    </div>
  );
}
