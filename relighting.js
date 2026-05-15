/**
 * Re-iluminação simples: um único ajuste global calculado no primeiro frame (V1).
 */
import sharp from 'sharp';

/**
 * @param {Buffer} firstForegroundRgba
 * @param {Buffer} backgroundRgb - fundo já redimensionado (ex.: 960×540)
 * @returns {Promise<{ brightness: number; saturation: number }>}
 */
export async function computeGlobalRelight(firstForegroundRgba, backgroundRgb) {
  const fgStats = await sharp(firstForegroundRgba).stats();
  const bgStats = await sharp(backgroundRgb).stats();

  const mean = (stats) =>
    stats.channels
      .filter((c) => c && typeof c.mean === 'number')
      .slice(0, 3)
      .reduce((s, c) => s + c.mean, 0) / 3;

  const fgL = mean(fgStats);
  const bgL = mean(bgStats);
  const fgSafe = Number.isFinite(fgL) ? fgL : 128;
  const bgSafe = Number.isFinite(bgL) ? bgL : 128;
  const ratio = bgSafe / Math.max(fgSafe, 1);

  const brightness = Math.min(1.25, Math.max(0.72, Number.isFinite(ratio) ? ratio : 1));
  const satRaw = 0.95 + (bgSafe - fgSafe) / 400;
  const saturation = Math.min(1.15, Math.max(0.9, Number.isFinite(satRaw) ? satRaw : 1));

  return { brightness, saturation };
}

/**
 * Referência visual: extrai tendência de temperatura e intensidade de luz.
 * Não é PBR completo, mas estabiliza tom/energia no personagem.
 * @param {Buffer} referenceRgb
 */
export async function analyzeReferenceLighting(referenceRgb) {
  const stats = await sharp(referenceRgb).stats();
  const [r, g, b] = stats.channels.slice(0, 3).map((c) => {
    const m = c?.mean;
    return Number.isFinite(m) ? m : 0;
  });
  const lum = (r + g + b) / 3;
  const warmth = (r - b) / 255;
  const lumSafe = Number.isFinite(lum) ? lum : 128;
  const warmthSafe = Number.isFinite(warmth) ? warmth : 0;
  return {
    intensity: Math.min(1.25, Math.max(0.75, lumSafe / 128)),
    warmth: Math.min(0.18, Math.max(-0.18, warmthSafe)),
  };
}

/**
 * @param {Buffer} foregroundRgba
 * @param {{ brightness: number; saturation: number }} params
 */
export async function applyGlobalRelight(foregroundRgba, params) {
  const br = Number(params?.brightness);
  const sat = Number(params?.saturation);
  const inten = Number(params?.intensity);
  const brightness = (Number.isFinite(br) ? br : 1) * (Number.isFinite(inten) ? inten : 1);
  const saturation = Number.isFinite(sat) ? sat : 1;

  // Apenas modulate: o composite `screen` com camada RGB sobre RGBA pode zerar/corromper o alpha
  // no libvips, deixando só o fundo (quase preto se a REF for escura).
  return sharp(foregroundRgba)
    .ensureAlpha()
    .modulate({
      brightness: Number.isFinite(brightness) ? brightness : 1,
      saturation: Number.isFinite(saturation) ? saturation : 1,
    })
    .png()
    .toBuffer();
}
