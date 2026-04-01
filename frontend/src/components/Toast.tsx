import { useEffect, useRef } from 'react';

interface ToastProps {
  message: string;
  type: 'success' | 'error';
  onClose: () => void;
}

export default function Toast({ message, type, onClose }: ToastProps) {
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    timer.current = setTimeout(onClose, 3000);
    return () => { if (timer.current) clearTimeout(timer.current); };
  }, [onClose]);

  return (
    <div className={`toast toast-${type}`}>
      <span className="mr-2">{type === 'success' ? '✓' : '✕'}</span>
      {message}
    </div>
  );
}
