/**
 * Re-injeção de alta frequência após re-iluminação global (Sharp modulate):
 * preserva textura / granulometria da câmara no recorte, aplicando só a camada
 * baixa-frequência do resultado relightado.
 *
 * Implementação: separação tipo Laplaciano / frequency split em espaço pré-multiplicado
 * (reduz halos em bordas com alpha).
 */
import sharp from 'sharp';

const DEFAULTS = {
  /** Sigma Gaussian (Sharp blur) — valores ~1–3 para 540p–1080p */
  sigma: 2.2,
  /** Peso do detalhe de alta frequência do original (0 = ignorar HF, 1 = máximo) */
  detailStrength: 0.92,
  /** Alpha mínimo (0–255) para aplicar mistura; abaixo usa só relit */
  alphaMin: 10,
};

/**
 * @param {Buffer} originalRgbaPng — RGBA antes do modulate
 * @param {Buffer} relitRgbaPng — RGBA depois do modulate
 * @param {{ sigma?: number; detailStrength?: number; alphaMin?: number }} [opts]
 * @returns {Promise<Buffer>} PNG RGBA
 */
export async function mergeRelightPreserveHighFreq(originalRgbaPng, relitRgbaPng, opts = {}) {
  const sigma = Number.isFinite(opts.sigma) ? opts.sigma : DEFAULTS.sigma;
  const detailStrength = Number.isFinite(opts.detailStrength) ? opts.detailStrength : DEFAULTS.detailStrength;
  const alphaMin = Number.isFinite(opts.alphaMin) ? opts.alphaMin : DEFAULTS.alphaMin;

  const oMeta = await sharp(originalRgbaPng).metadata();
  const rMeta = await sharp(relitRgbaPng).metadata();
  const w = oMeta.width || rMeta.width;
  const h = oMeta.height || rMeta.height;
  if (!w || !h) {
    return relitRgbaPng;
  }

  const orig = await sharp(originalRgbaPng)
    .resize(w, h, { fit: 'fill' })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const rel = await sharp(relitRgbaPng)
    .resize(w, h, { fit: 'fill' })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  if (orig.info.channels < 4 || rel.info.channels < 4) {
    return relitRgbaPng;
  }

  const px = w * h;
  const premulOrig = Buffer.allocUnsafe(px * 3);
  const premulRel = Buffer.allocUnsafe(px * 3);

  for (let i = 0; i < px; i += 1) {
    const p = i * 4;
    const ao = orig.data[p + 3] / 255;
    const ar = rel.data[p + 3] / 255;
    premulOrig[i * 3] = orig.data[p] * ao;
    premulOrig[i * 3 + 1] = orig.data[p + 1] * ao;
    premulOrig[i * 3 + 2] = orig.data[p + 2] * ao;
    premulRel[i * 3] = rel.data[p] * ar;
    premulRel[i * 3 + 1] = rel.data[p + 1] * ar;
    premulRel[i * 3 + 2] = rel.data[p + 2] * ar;
  }

  const lowOrig = await sharp(premulOrig, { raw: { width: w, height: h, channels: 3 } })
    .blur(sigma)
    .raw()
    .toBuffer();

  const lowRel = await sharp(premulRel, { raw: { width: w, height: h, channels: 3 } })
    .blur(sigma)
    .raw()
    .toBuffer();

  const out = Buffer.allocUnsafe(px * 4);
  for (let i = 0; i < px; i += 1) {
    const p = i * 4;
    const aRel = rel.data[p + 3];
    const hi0 = premulOrig[i * 3] - lowOrig[i * 3];
    const hi1 = premulOrig[i * 3 + 1] - lowOrig[i * 3 + 1];
    const hi2 = premulOrig[i * 3 + 2] - lowOrig[i * 3 + 2];

    let pr = lowRel[i * 3] + detailStrength * hi0;
    let pg = lowRel[i * 3 + 1] + detailStrength * hi1;
    let pb = lowRel[i * 3 + 2] + detailStrength * hi2;

    if (aRel < alphaMin) {
      out[p] = rel.data[p];
      out[p + 1] = rel.data[p + 1];
      out[p + 2] = rel.data[p + 2];
      out[p + 3] = aRel;
      continue;
    }

    const a = aRel / 255;
    const inv = 1 / Math.max(a, 1e-4);
    pr = Math.min(255, Math.max(0, pr * inv));
    pg = Math.min(255, Math.max(0, pg * inv));
    pb = Math.min(255, Math.max(0, pb * inv));

    out[p] = Math.round(pr);
    out[p + 1] = Math.round(pg);
    out[p + 2] = Math.round(pb);
    out[p + 3] = aRel;
  }

  return sharp(out, { raw: { width: w, height: h, channels: 4 } }).png().toBuffer();
}
