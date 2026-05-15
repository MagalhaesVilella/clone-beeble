import { useEffect, useMemo, useState } from 'react';

const MAX_NATIVE_FRAMES = 240;

function longEdgeForResolution(res) {
  if (res === '540p') return 960;
  if (res === '720p') return 1280;
  return 1920;
}

function computeOutFrameSize(srcW, srcH, resKey) {
  const le = longEdgeForResolution(resKey);
  const sw = Math.max(1, Math.floor(srcW || 1));
  const sh = Math.max(1, Math.floor(srcH || 1));
  const scale = Math.min(le / sw, le / sh);
  return {
    width: Math.max(2, Math.floor((sw * scale) / 2) * 2),
    height: Math.max(2, Math.floor((sh * scale) / 2) * 2),
  };
}

function extractFpsForQuality(q) {
  if (q === 'fast') return 1;
  if (q === 'precise') return 4;
  return 2;
}

function formatTime(sec) {
  if (!Number.isFinite(sec)) return '0:00';
  const s = Math.max(0, sec);
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  const cs = Math.floor((s % 1) * 1000);
  return `${m}:${String(r).padStart(2, '0')}.${String(cs).padStart(3, '0')}`;
}

export default function App() {
  const [apiOk, setApiOk] = useState(null);
  const [pythonOk, setPythonOk] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [bgFile, setBgFile] = useState(null);
  const [bgUrl, setBgUrl] = useState('');
  const [resultUrl, setResultUrl] = useState('');
  const [prompt, setPrompt] = useState('');
  const [autopilot, setAutopilot] = useState(true);
  const [resolution, setResolution] = useState('1080p');
  const [centerTab, setCenterTab] = useState('history');
  const [progress, setProgress] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [status, setStatus] = useState('');
  const [lastJob, setLastJob] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);
  const [maskPreviewUrl, setMaskPreviewUrl] = useState('');
  const [sourceError, setSourceError] = useState('');
  const [maskMode, setMaskMode] = useState('auto');
  const [maskFile, setMaskFile] = useState(null);
  const [selectPointsNorm, setSelectPointsNorm] = useState([]);
  const [quality, setQuality] = useState('fast');
  const [videoProbe, setVideoProbe] = useState(null);
  const [probeError, setProbeError] = useState('');
  const [clipStartSec, setClipStartSec] = useState(0);
  const [clipEndSec, setClipEndSec] = useState(0);
  const [clipConfirmed, setClipConfirmed] = useState(false);
  const [maskPreviewBusy, setMaskPreviewBusy] = useState(false);

  const clipDur = useMemo(
    () => (Number.isFinite(clipEndSec) && Number.isFinite(clipStartSec) ? Math.max(0, clipEndSec - clipStartSec) : 0),
    [clipStartSec, clipEndSec],
  );
  const approxNativeFrames = useMemo(() => {
    if (!videoProbe?.nativeFps || clipDur <= 0) return 0;
    return Math.ceil(clipDur * videoProbe.nativeFps);
  }, [clipDur, videoProbe]);
  const approxExtractFrames = useMemo(() => {
    if (clipDur <= 0) return 0;
    return Math.ceil(clipDur * extractFpsForQuality(quality));
  }, [clipDur, quality]);

  const outFrameDims = useMemo(() => {
    if (!videoProbe?.width || !videoProbe?.height) return { width: 960, height: 540 };
    return computeOutFrameSize(videoProbe.width, videoProbe.height, resolution);
  }, [videoProbe, resolution]);

  const playerAspectStyle = useMemo(() => {
    const w = lastJob?.outWidth || outFrameDims.width;
    const h = lastJob?.outHeight || outFrameDims.height;
    if (w > 0 && h > 0) return { aspectRatio: `${w} / ${h}` };
    return { aspectRatio: '16 / 9' };
  }, [lastJob, outFrameDims]);

  const clipValid =
    videoProbe &&
    clipDur >= 0.04 &&
    clipStartSec >= 0 &&
    clipEndSec <= videoProbe.duration + 0.001 &&
    clipEndSec > clipStartSec &&
    approxNativeFrames <= MAX_NATIVE_FRAMES;

  const maskOk = maskMode !== 'upload' || maskFile;
  const pythonReady = maskMode === 'fill' || pythonOk === true;
  const canProcess =
    videoFile &&
    bgFile &&
    maskOk &&
    !processing &&
    apiOk === true &&
    pythonReady &&
    !!videoProbe &&
    clipConfirmed &&
    clipValid;

  useEffect(() => {
    let c = false;
    let attempts = 0;
    const maxAttempts = 120;

    const poll = () => {
      if (c) return;
      attempts += 1;
      fetch('/api/health')
        .then((r) => r.json())
        .then((d) => {
          if (c) return;
          setApiOk(!!d?.ok);
          const py = d?.python?.ok;
          setPythonOk(py === true ? true : py === false ? false : null);

          const apiUp = !!d?.ok;
          const pythonDone = py === true || maskMode === 'fill';
          if (apiUp && pythonDone) return;
          if (attempts >= maxAttempts) return;
          window.setTimeout(poll, apiUp ? 1200 : 600);
        })
        .catch(() => {
          if (c) return;
          setApiOk(false);
          setPythonOk(null);
          if (attempts < maxAttempts) window.setTimeout(poll, 600);
        });
    };

    poll();
    return () => {
      c = true;
    };
  }, [maskMode]);

  useEffect(() => {
    if (!videoFile) {
      setVideoProbe(null);
      setProbeError('');
      setClipConfirmed(false);
      setClipStartSec(0);
      setClipEndSec(0);
      return;
    }
    let cancelled = false;
    const fd = new FormData();
    fd.append('video', videoFile);
    setProbeError('');
    setVideoProbe(null);
    setClipConfirmed(false);
    fetch('/api/switchx/probe', { method: 'POST', body: fd })
      .then(async (r) => {
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
        return data;
      })
      .then((data) => {
        if (cancelled) return;
        setVideoProbe(data);
        const end = Math.min(data.duration, data.maxSegmentSec ?? data.duration);
        setClipStartSec(0);
        setClipEndSec(end);
      })
      .catch((e) => {
        if (!cancelled) setProbeError(e.message || 'Falha ao analisar o vídeo');
      });
    return () => {
      cancelled = true;
    };
  }, [videoFile]);

  useEffect(() => {
    setMaskPreviewUrl('');
  }, [clipStartSec, clipEndSec]);

  const handlePreviewMask = async () => {
    if (!videoFile || !clipConfirmed || !clipValid) return;
    if (maskMode !== 'fill' && pythonOk !== true) return;
    setMaskPreviewBusy(true);
    setStatus('');
    try {
      const fd = new FormData();
      fd.append('video', videoFile);
      fd.append('maskMode', maskMode);
      fd.append('outputResolution', resolution);
      fd.append('clipStartSec', String(clipStartSec));
      fd.append('clipEndSec', String(clipEndSec));
      if (maskMode === 'upload' && maskFile) fd.append('maskUpload', maskFile);
      if (maskMode === 'select' && selectPointsNorm.length > 0) {
        const pixels = selectPointsNorm.map(({ nx, ny }) => ({
          x: Math.round(Math.min(1, Math.max(0, nx)) * Math.max(0, outFrameDims.width - 1)),
          y: Math.round(Math.min(1, Math.max(0, ny)) * Math.max(0, outFrameDims.height - 1)),
        }));
        fd.append('maskPoints', JSON.stringify(pixels));
      }
      const res = await fetch('/api/switchx/preview-mask', { method: 'POST', body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      const u = data.maskPreviewUrl || '';
      setMaskPreviewUrl(u ? `${u}?t=${Date.now()}` : '');
      setStatus('Máscara pré-visualizada (1.º frame do trecho).');
    } catch (e) {
      setStatus(`Erro máscara: ${e.message}`);
    } finally {
      setMaskPreviewBusy(false);
    }
  };

  const handleProcess = async () => {
    if (!canProcess) return;
    setProcessing(true);
    setProgress(8);
    setStatus('A enviar…');
    setResultUrl('');
    setMaskPreviewUrl('');
    try {
      const fd = new FormData();
      fd.append('video', videoFile);
      fd.append('background', bgFile);
      if (referenceFile) fd.append('reference', referenceFile);
      fd.append('maskMode', maskMode);
      if (maskMode === 'upload' && maskFile) fd.append('maskUpload', maskFile);
      if (maskMode === 'select' && selectPointsNorm.length > 0) {
        const pixels = selectPointsNorm.map(({ nx, ny }) => ({
          x: Math.round(Math.min(1, Math.max(0, nx)) * Math.max(0, outFrameDims.width - 1)),
          y: Math.round(Math.min(1, Math.max(0, ny)) * Math.max(0, outFrameDims.height - 1)),
        }));
        fd.append('maskPoints', JSON.stringify(pixels));
      }
      fd.append('model', 'google');
      fd.append('quality', quality);
      fd.append('outputResolution', resolution);
      fd.append('clipStartSec', String(clipStartSec));
      fd.append('clipEndSec', String(clipEndSec));
      setProgress(20);
      setStatus('A processar (Google Auto)…');
      const res = await fetch('/api/switchx', { method: 'POST', body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const errMsg = data.error || `Erro HTTP ${res.status}`;
        throw new Error(res.status === 503 ? `Serviço indisponível: ${errMsg}` : errMsg);
      }
      setProgress(100);
      const t = Date.now();
      const url = data.videoUrl || '';
      const bust = url ? `${url}${url.includes('?') ? '&' : '?'}t=${t}` : '';
      const mp = data.maskPreviewUrl || '';
      const maskBust = mp ? `${mp}${mp.includes('?') ? '&' : '?'}t=${t}` : '';
      setResultUrl(bust);
      setMaskPreviewUrl(maskBust);
      setStatus(`Concluído · ${data.frames ?? '?'} frames · ~${data.durationApprox?.toFixed(2) ?? '?'}s`);
      setLastJob({
        id: Date.now(),
        resultUrl: bust,
        videoUrl,
        bgUrl,
        frames: data.frames,
        fps: data.fps,
        durationApprox: data.durationApprox,
        prompt: prompt || 'Troca de fundo local (clone SwitchX V1).',
        date: new Date().toLocaleDateString('pt-PT'),
        maskPreviewUrl: maskBust,
        maskMode: data.maskMode || maskMode,
        outWidth: data.outWidth,
        outHeight: data.outHeight,
      });
    } catch (e) {
      setStatus(`Erro: ${e.message}`);
      setProgress(0);
    } finally {
      setProcessing(false);
    }
  };

  const displayVideo = resultUrl || videoUrl;
  const maskReady = !!videoUrl;
  /** No player principal: sobrepor tint da máscara (1.º frame) ao source até existir output final. */
  const showMaskOnMainPlayer = Boolean(!resultUrl && maskPreviewUrl && videoUrl);

  return (
    <div className="sw-root">
      <aside className="sw-rail">
        <div className="sw-rail-icons">
          <span className="sw-ico active" title="Home" />
          <span className="sw-ico" title="Upload" />
          <span className="sw-ico" title="Mask" />
          <span className="sw-ico" title="History" />
        </div>
        <div className="sw-rail-inner">
          <div className="sw-brand-row">
            <span className="sw-brand">SWITCHX</span>
            <a className="sw-docs" href="#" onClick={(e) => e.preventDefault()}>
              View Docs
            </a>
          </div>

          <section className="sw-sec">
            <div className="sw-sec-h">
              <span className="sw-num">1</span>
              <h2>Source</h2>
            </div>
            <p className="sw-sub">Trecho até {MAX_NATIVE_FRAMES} frames (nativo) · vídeo até 10 min</p>
            <label className="sw-upload">
              <input
                type="file"
                accept="video/mp4,video/quicktime,image/png,image/jpeg,image/jpg"
                className="sw-hide"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  setSourceError('');
                  if (f && !f.type.startsWith('video/')) {
                    setVideoFile(null);
                    setVideoUrl('');
                    setSourceError('O Source do modo SwitchX deve ser um vídeo (mp4/mov).');
                    return;
                  }
                  setVideoFile(f || null);
                  if (videoUrl) URL.revokeObjectURL(videoUrl);
                  setVideoUrl(f ? URL.createObjectURL(f) : '');
                  setMaskPreviewUrl('');
                }}
              />
              Upload
            </label>
            {sourceError && <p className="sw-err">{sourceError}</p>}
            {videoUrl && (
              <div className="sw-thumb">
                <video src={videoUrl} muted playsInline className="sw-thumb-v" />
              </div>
            )}
            {probeError && <p className="sw-err">{probeError}</p>}
            {videoProbe && (
              <div className="sw-clip-panel">
                <div className="sw-clip-title">Definir trecho do vídeo</div>
                <div className="sw-clip-row">
                  <label>
                    Início ({formatTime(clipStartSec)})
                    <input
                      type="range"
                      min={0}
                      max={Math.max(0, videoProbe.duration - 0.05)}
                      step={0.01}
                      value={Math.min(clipStartSec, videoProbe.duration - 0.05)}
                      onChange={(e) => {
                        const v = parseFloat(e.target.value);
                        setClipStartSec(v);
                        setClipEndSec((end) => (end <= v + 0.04 ? Math.min(videoProbe.duration, v + 0.1) : end));
                        setClipConfirmed(false);
                      }}
                    />
                  </label>
                </div>
                <div className="sw-clip-row">
                  <label>
                    Fim ({formatTime(clipEndSec)})
                    <input
                      type="range"
                      min={Math.min(videoProbe.duration, clipStartSec + 0.04)}
                      max={videoProbe.duration}
                      step={0.01}
                      value={Math.max(clipStartSec + 0.04, Math.min(clipEndSec, videoProbe.duration))}
                      onChange={(e) => {
                        setClipEndSec(parseFloat(e.target.value));
                        setClipConfirmed(false);
                      }}
                    />
                  </label>
                </div>
                <div className="sw-clip-grid">
                  <label>
                    Início (s)
                    <input
                      type="number"
                      step="0.01"
                      min={0}
                      max={videoProbe.duration}
                      value={Number.isFinite(clipStartSec) ? clipStartSec : 0}
                      onChange={(e) => {
                        const v = Math.max(0, parseFloat(e.target.value) || 0);
                        setClipStartSec(v);
                        setClipConfirmed(false);
                      }}
                    />
                  </label>
                  <label>
                    Fim (s)
                    <input
                      type="number"
                      step="0.01"
                      min={0}
                      max={videoProbe.duration}
                      value={Number.isFinite(clipEndSec) ? clipEndSec : 0}
                      onChange={(e) => {
                        const v = Math.min(videoProbe.duration, parseFloat(e.target.value) || 0);
                        setClipEndSec(v);
                        setClipConfirmed(false);
                      }}
                    />
                  </label>
                </div>
                <div className="sw-clip-grid">
                  <label>
                    Frame inicial (~)
                    <input
                      type="number"
                      min={0}
                      readOnly
                      value={Math.max(0, Math.floor(clipStartSec * videoProbe.nativeFps))}
                    />
                  </label>
                  <label>
                    Frame final (~)
                    <input
                      type="number"
                      min={0}
                      readOnly
                      value={Math.max(0, Math.ceil(clipEndSec * videoProbe.nativeFps) - 1)}
                    />
                  </label>
                </div>
                <p className="sw-clip-meta">
                  Duração: {clipDur.toFixed(2)}s · ~{approxNativeFrames} frames nativos @ {videoProbe.nativeFps.toFixed(2)}{' '}
                  fps · ~{approxExtractFrames} frames extraídos ({extractFpsForQuality(quality)} fps) · canvas{' '}
                  {outFrameDims.width}×{outFrameDims.height}px ({resolution})
                </p>
                <p className="sw-clip-meta">Máximo: {MAX_NATIVE_FRAMES} frames nativos</p>
                {!clipValid && videoProbe && (
                  <p className="sw-err">
                    {approxNativeFrames > MAX_NATIVE_FRAMES
                      ? `Trecho longo demais (~${approxNativeFrames} > ${MAX_NATIVE_FRAMES}).`
                      : 'Ajusta início e fim.'}
                  </p>
                )}
                <button
                  type="button"
                  className="sw-clip-confirm"
                  disabled={!clipValid}
                  onClick={() => setClipConfirmed(true)}
                >
                  {clipConfirmed ? 'Trecho confirmado ✓' : 'Confirmar trecho'}
                </button>
              </div>
            )}
          </section>

          <section className="sw-sec">
            <div className="sw-sec-h">
              <span className="sw-num">2</span>
              <h2>Mask</h2>
              <a className="sw-guide" href="#" onClick={(e) => e.preventDefault()}>
                Guide
              </a>
            </div>
            <div className="sw-tabs">
              {['auto', 'select', 'fill', 'upload'].map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={maskMode === mode ? 'sw-tab on' : 'sw-tab'}
                  onClick={() => {
                    setMaskMode(mode);
                    if (mode !== 'upload') setMaskFile(null);
                    if (mode !== 'select') setSelectPointsNorm([]);
                  }}
                >
                  {mode[0].toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>
            {maskMode === 'upload' && (
              <label className="sw-upload" style={{ marginTop: 8 }}>
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  className="sw-hide"
                  onChange={(e) => setMaskFile(e.target.files?.[0] || null)}
                />
                Máscara (PNG/JPG, canal L / alpha)
              </label>
            )}
            <div className="sw-mask-preview">
              {maskReady ? (
                maskMode === 'select' ? (
                  <div className="sw-mask-stack sw-select-hit">
                    <video src={videoUrl} muted playsInline className="sw-mask-v" />
                    <div
                      className="sw-select-overlay"
                      role="presentation"
                      onClick={(e) => {
                        const el = e.currentTarget;
                        const rect = el.getBoundingClientRect();
                        const nx = (e.clientX - rect.left) / rect.width;
                        const ny = (e.clientY - rect.top) / rect.height;
                        setSelectPointsNorm((prev) => {
                          if (prev.length >= 10) return prev;
                          return [...prev, { nx, ny }];
                        });
                      }}
                    >
                      {selectPointsNorm.map((p, i) => (
                        <span
                          key={i}
                          className="sw-select-dot"
                          style={{ left: `${p.nx * 100}%`, top: `${p.ny * 100}%` }}
                        />
                      ))}
                    </div>
                    <div className="sw-select-hint">
                      Até 10 cliques · canvas = {outFrameDims.width}×{outFrameDims.height}px
                      <button type="button" className="sw-select-clear" onClick={() => setSelectPointsNorm([])}>
                        Limpar pontos
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="sw-mask-stack">
                    <video src={videoUrl} muted playsInline className="sw-mask-v" />
                  </div>
                )
              ) : (
                <span className="sw-ph">Upload media first</span>
              )}
            </div>
            <div className="sw-res">
              <label>
                Mask quality
                <select value={quality} onChange={(e) => setQuality(e.target.value)}>
                  <option value="fast">Fast</option>
                  <option value="standard">Standard</option>
                  <option value="precise">Precise</option>
                </select>
              </label>
            </div>
            <button
              type="button"
              className="sw-btn-preview"
              disabled={
                !videoFile ||
                !clipConfirmed ||
                !clipValid ||
                maskPreviewBusy ||
                (maskMode !== 'fill' && pythonOk !== true)
              }
              onClick={handlePreviewMask}
            >
              {maskPreviewBusy ? 'A gerar máscara…' : 'Pré-visualizar máscara (1.º frame)'}
            </button>
            {maskPreviewUrl && <img src={maskPreviewUrl} alt="Mask preview" className="sw-ref-img" />}
          </section>

          <section className="sw-sec">
            <div className="sw-sec-h">
              <span className="sw-num">3</span>
              <h2>Fundo (composição)</h2>
            </div>
            <div className="sw-ref">
              {bgUrl ? (
                <img src={bgUrl} alt="" className="sw-ref-img" />
              ) : (
                <p className="sw-ph">Novo fundo — composição final (JPG/PNG)</p>
              )}
              <div className="sw-ref-btns">
                <button type="button" className="sw-btn-ai" disabled>
                  Create with AI
                </button>
                <label className="sw-btn-up">
                  Upload
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    className="sw-hide"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      setBgFile(f || null);
                      setReferenceFile(f || null);
                      if (bgUrl) URL.revokeObjectURL(bgUrl);
                      setBgUrl(f ? URL.createObjectURL(f) : '');
                    }}
                  />
                </label>
              </div>
            </div>
            <div className="sw-prompt-row">
              <span>Prompt</span>
              <label className="sw-ap">
                <input type="checkbox" checked={autopilot} onChange={(e) => setAutopilot(e.target.checked)} />
                Auto pilot
              </label>
            </div>
            <textarea
              className="sw-ta"
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={autopilot ? 'Descreva o cenário (opcional nesta V1).' : 'Prompt manual…'}
            />
            <div className="sw-res">
              <label>
                Resolução
                <select
                  value={resolution}
                  onChange={(e) => {
                    setResolution(e.target.value);
                    setMaskPreviewUrl('');
                  }}
                >
                  <option value="540p">540p</option>
                  <option value="720p">720p</option>
                  <option value="1080p">1080p</option>
                </select>
              </label>
              <label>
                Modelo
                <select value="google" disabled>
                  <option value="google">Google</option>
                </select>
              </label>
            </div>
            <button type="button" className="sw-gen" disabled={!canProcess} onClick={handleProcess}>
              {processing ? `Processing ${progress}%…` : 'Generate'}
            </button>
            {(processing || progress > 0) && (
              <div className="sw-bar">
                <div style={{ width: `${progress}%` }} />
              </div>
            )}
            {apiOk === false && (
              <p className="sw-err">API offline — corre <code>npm run dev</code> na pasta beeble-clone.</p>
            )}
            {apiOk === true && pythonOk === false && maskMode !== 'fill' && (
              <p className="sw-err">
                Worker Python indisponível (segmentação). Instala dependências na pasta <code>backend</code>:{' '}
                <code>pip install -r requirements.txt</code> e reinicia a API.
              </p>
            )}
            {status && <p className="sw-st">{status}</p>}
          </section>
        </div>
      </aside>

      <main className="sw-main">
        <nav className="sw-main-tabs">
          {[
            ['history', 'HISTORY'],
            ['showcase', 'SHOWCASE'],
            ['tutorial', 'TUTORIAL'],
          ].map(([id, label]) => (
            <button key={id} type="button" className={centerTab === id ? 'on' : ''} onClick={() => setCenterTab(id)}>
              {label}
            </button>
          ))}
        </nav>
        <div className="sw-player-wrap">
          {displayVideo ? (
            <div className={`sw-player${showMaskOnMainPlayer ? ' sw-player-masked' : ''}`} style={playerAspectStyle}>
              <video
                src={showMaskOnMainPlayer ? videoUrl : displayVideo}
                controls
                playsInline
                className="sw-player-v"
              />
              {showMaskOnMainPlayer && (
                <div
                  className="sw-player-mask-tint"
                  style={{
                    WebkitMaskImage: `url(${maskPreviewUrl})`,
                    maskImage: `url(${maskPreviewUrl})`,
                  }}
                  aria-hidden
                />
              )}
              {showMaskOnMainPlayer && (
                <span className="sw-player-mask-badge">Máscara · 1.º frame do trecho</span>
              )}
            </div>
          ) : (
            <div className="sw-player empty" style={videoProbe ? playerAspectStyle : undefined}>
              <span>Pré-visualização</span>
            </div>
          )}
        </div>
        <div className="sw-strip">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="sw-strip-cell">
              {displayVideo ? (
                <video
                  src={displayVideo}
                  muted
                  loop
                  autoPlay
                  playsInline
                  className="sw-strip-v"
                />
              ) : (
                <span className="sw-ph">—</span>
              )}
            </div>
          ))}
        </div>
      </main>

      <aside className="sw-hist">
        <div className="sw-hist-head">
          <span className="sw-hist-title">Jobs</span>
          <span className="sw-hist-count">{lastJob ? 1 : 0}</span>
        </div>
        {lastJob ? (
          <article className="sw-card">
            <header className="sw-card-h">
              <span className="sw-card-title">Último output</span>
              <time>{lastJob.date}</time>
            </header>
            <p className="sw-meta">
              {lastJob.frames} frames · {lastJob.fps} fps · ~{lastJob.durationApprox?.toFixed(2)}s
            </p>
            <div className="sw-quad">
              <div>
                <small>OUTPUT</small>
                {lastJob.resultUrl ? (
                  <video src={lastJob.resultUrl} muted loop autoPlay playsInline className="sq-v" />
                ) : (
                  <div className="sq-ph" />
                )}
              </div>
              <div>
                <small>SOURCE</small>
                {lastJob.videoUrl ? (
                  <video src={lastJob.videoUrl} muted loop autoPlay playsInline className="sq-v" />
                ) : (
                  <div className="sq-ph" />
                )}
              </div>
              <div>
                <small>ALPHA</small>
                {lastJob.maskPreviewUrl ? <img src={lastJob.maskPreviewUrl} alt="" className="sq-img" /> : <div className="sq-alpha">α</div>}
              </div>
              <div>
                <small>REF</small>
                {lastJob.bgUrl ? <img src={lastJob.bgUrl} alt="" className="sq-img" /> : <div className="sq-ph" />}
              </div>
            </div>
            <p className="sw-card-prompt">{lastJob.prompt}</p>
          </article>
        ) : (
          <p className="sw-hist-empty">O histórico aparece após gerar.</p>
        )}
      </aside>
    </div>
  );
}
