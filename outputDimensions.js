/**
 * Tamanho do canvas de processamento: encaixa o vídeo dentro de uma caixa
 * com o maior lado = longEdge (px), preservando proporção (pares para H.264).
 */
export function longEdgeForResolution(res) {
  const r = String(res || '1080p').toLowerCase();
  if (r === '540p') return 960;
  if (r === '720p') return 1280;
  return 1920;
}

export function computeOutputFrameSize(srcW, srcH, resolutionKey) {
  const longEdge = longEdgeForResolution(resolutionKey);
  const sw = Math.max(1, Math.floor(Number(srcW) || 1));
  const sh = Math.max(1, Math.floor(Number(srcH) || 1));
  const scale = Math.min(longEdge / sw, longEdge / sh);
  const width = Math.max(2, Math.floor((sw * scale) / 2) * 2);
  const height = Math.max(2, Math.floor((sh * scale) / 2) * 2);
  return { width, height };
}
