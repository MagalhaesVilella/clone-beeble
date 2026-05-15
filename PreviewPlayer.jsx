export default function PreviewPlayer({ originalUrl, backgroundUrl, resultUrl }) {
  return (
    <div className="preview-grid">
      <div className="pv-card">
        <h3>Original</h3>
        {originalUrl ? <video src={originalUrl} controls muted className="pv" /> : <p className="muted">—</p>}
      </div>
      <div className="pv-card">
        <h3>Fundo</h3>
        {backgroundUrl ? <img src={backgroundUrl} alt="" className="pv img" /> : <p className="muted">—</p>}
      </div>
      <div className="pv-card wide">
        <h3>Resultado</h3>
        {resultUrl ? (
          <video src={resultUrl} controls className="pv" />
        ) : (
          <p className="muted">O vídeo processado aparece aqui.</p>
        )}
      </div>
    </div>
  );
}
