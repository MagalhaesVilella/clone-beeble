export default function ProcessButton({ disabled, processing, progress, onClick }) {
  return (
    <div className="process-wrap">
      <button type="button" className="btn-process" disabled={disabled} onClick={onClick}>
        {processing ? 'A processar…' : '🎬 Processar'}
      </button>
      {(processing || progress > 0) && (
        <div className="bar">
          <div className="fill" style={{ width: `${Math.min(100, progress)}%` }} />
        </div>
      )}
    </div>
  );
}
