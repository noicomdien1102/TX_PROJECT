import { useCallback, useEffect, useState } from 'react';
import { api, type PredictResponse, type Side } from './api';
import Grid from './components/Grid';
import ControlPanel from './components/ControlPanel';
import Dashboard from './components/Dashboard';
import Toast from './components/Toast';

interface ToastState {
  message: string;
  type: 'success' | 'error';
  id: number;
}

function App() {
  const [gridCells, setGridCells] = useState<Side[]>([]);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<ToastState | null>(null);

  const showToast = useCallback((message: string, type: 'success' | 'error') => {
    setToast({ message, type, id: Date.now() });
  }, []);

  const fetchPrediction = useCallback(async () => {
    try {
      const data = await api.predict();
      setPrediction(data);
      // Sync grid from tail (up to 100) — derive from total for display
      if (data.sequence_tail) {
        setGridCells(data.sequence_tail.slice(-100) as Side[]);
      }
    } catch {
      // silently ignore on initial load when history is empty
    }
  }, []);

  useEffect(() => {
    fetchPrediction();
  }, [fetchPrediction]);

  const handleDiceAdded = () => fetchPrediction();
  const handleManualAdded = () => fetchPrediction();

  const handleImageUploaded = (results: string[]) => {
    setGridCells(results.slice(0, 100) as Side[]);
    fetchPrediction();
  };

  const handleReset = () => {
    setGridCells([]);
    setPrediction(null);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur-xl bg-[rgba(10,14,26,0.85)] border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center text-lg shadow-lg shadow-indigo-500/30">
              🎯
            </div>
            <div>
              <h1 className="text-base font-bold text-white tracking-tight">TX Prediction Engine</h1>
              <p className="text-[11px] text-slate-500">Weighted Markov Chain · Cầu Detection · M4 Optimized</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span
              className={`w-2 h-2 rounded-full inline-block ${
                prediction ? 'bg-emerald-400 pulse-dot' : 'bg-slate-600'
              }`}
            />
            {prediction ? `${prediction.total_entries} bản ghi` : 'Chưa có dữ liệu'}
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            <Grid cells={gridCells} />
            <ControlPanel
              onDiceAdded={handleDiceAdded}
              onManualAdded={handleManualAdded}
              onImageUploaded={handleImageUploaded}
              onReset={handleReset}
              onToast={showToast}
              loading={loading}
              setLoading={setLoading}
            />
          </div>

          {/* Right Column — Dashboard */}
          <div className="xl:h-fit xl:sticky xl:top-24">
            <Dashboard data={prediction} loading={loading} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center py-4 text-xs text-slate-600 border-t border-white/5">
        TX Prediction Engine · FastAPI + React + Vite · M4
      </footer>

      {/* Toast */}
      {toast && (
        <Toast
          key={toast.id}
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
}

export default App;
