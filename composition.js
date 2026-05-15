/**
 * Composição Sharp: recorte com pena, sombra elíptica, pessoa sobre fundo.
 * Ordem: frame PNG + alpha (L) → RGBA (fusão manual fiável) → fundo.
 */
import sharp from 'sharp';

const DEFAULT = { width: 960, height: 540 };

export function getOutputDimensions(dims) {
  return { width: dims?.width ?? DEFAULT.width, height: dims?.height ?? DEFAULT.height };
}

/**
 * Recorta a pessoa com alpha (borda suave). Fusão RGB+A em memória — evita bugs do joinChannel com PNG.
 */
export async function makeCutout(framePath, alpha, alphaW, alphaH, dims = DEFAULT) {
  const { width: ow, height: oh } = getOutputDimensions(dims);

  const normalizedRgb = await sharp(framePath)
    .resize(ow, oh, { fit: 'contain', position: 'centre', background: { r: 0, g: 0, b: 0 } })
    .toColourspace('srgb')
    .removeAlpha()
    .png()
    .toBuffer();

  const { data: rgbData, info } = await sharp(normalizedRgb)
    .raw()
    .toBuffer({ resolveWithObject: true });

  const ch = info.channels || 3;
  const pixels = ow * oh;
  const rgb = Buffer.allocUnsafe(pixels * 3);
  if (ch === 3 && rgbData.length >= pixels * 3) {
    rgbData.copy(rgb, 0, 0, pixels * 3);
  } else if (ch === 4) {
    for (let i = 0, p = 0; i < pixels; i += 1, p += 3) {
      const s = i * 4;
      rgb[p] = rgbData[s];
      rgb[p + 1] = rgbData[s + 1];
      rgb[p + 2] = rgbData[s + 2];
    }
  } else if (ch === 1) {
    for (let i = 0, p = 0; i < pixels; i += 1, p += 3) {
      const v = rgbData[i];
      rgb[p] = v;
      rgb[p + 1] = v;
      rgb[p + 2] = v;
    }
  } else if (ch === 2) {
    for (let i = 0, p = 0; i < pixels; i += 1, p += 3) {
      const s = i * 2;
      const v = rgbData[s];
      rgb[p] = v;
      rgb[p + 1] = v;
      rgb[p + 2] = v;
    }
  } else {
    throw new Error(`makeCutout: canais RGB inesperados (${ch}).`);
  }

  const alphaRawBuf = Buffer.from(alpha);
  const { data: aPre } = await sharp(alphaRawBuf, {
    raw: { width: alphaW, height: alphaH, channels: 1 },
  })
    .resize(ow, oh, { kernel: sharp.kernel.cubic })
    .raw()
    .toBuffer({ resolveWithObject: true });

  let sumPre = 0;
  for (let i = 0; i < pixels; i += 1) sumPre += aPre[i];
  const meanPre = sumPre / pixels;
  // Blur forte em máscaras já fracas espalha alpha e pode “apagar” a figura no vídeo final.
  const blurSigma = meanPre < 28 ? 0.9 : meanPre < 55 ? 1.8 : 2.8;
  const gamma = meanPre < 28 ? 1.0 : 1.05;

  const { data: aData } = await sharp(alphaRawBuf, {
    raw: { width: alphaW, height: alphaH, channels: 1 },
  })
    .resize(ow, oh, { kernel: sharp.kernel.cubic })
    .blur(blurSigma)
    .gamma(gamma)
    .raw()
    .toBuffer({ resolveWithObject: true });

  if (aData.length < pixels) {
    throw new Error('makeCutout: buffer alpha incompleto.');
  }

  let sumA = 0;
  for (let i = 0; i < pixels; i += 1) sumA += aData[i];
  const meanA = sumA / pixels;
  if (meanA < 8) {
    console.warn(
      `[switchx:composition] makeCutout: alpha médio muito baixo (${meanA.toFixed(2)}/255) — o recorte pode ficar invisível.`,
    );
  } else {
    console.log(
      `[switchx:composition] makeCutout: alpha médio=${meanA.toFixed(1)}/255 (pré-blur≈${meanPre.toFixed(1)}, blur=${blurSigma}, gamma=${gamma}, ${pixels}px)`,
    );
  }

  const out = Buffer.allocUnsafe(pixels * 4);
  for (let i = 0, p = 0, r = 0; i < pixels; i += 1, p += 4, r += 3) {
    out[p] = rgb[r];
    out[p + 1] = rgb[r + 1];
    out[p + 2] = rgb[r + 2];
    out[p + 3] = aData[i];
  }

  return sharp(out, { raw: { width: ow, height: oh, channels: 4 } }).png().toBuffer();
}

/**
 * Fundo redimensionado (buffer) para estatísticas e consistência.
 */
export async function getBackgroundLayer(bgPath, dims = DEFAULT) {
  const { width: ow, height: oh } = getOutputDimensions(dims);
  return sharp(bgPath).resize(ow, oh, { fit: 'cover' }).removeAlpha().png().toBuffer();
}

/**
 * Sombra + pessoa sobre fundo.
 */
export async function compositeCutoutOnBackground(bgLayerBuffer, cutoutRgba, dims = DEFAULT) {
  const { width: ow, height: oh } = getOutputDimensions(dims);

  const { data, info } = await sharp(cutoutRgba)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const ch = info.channels || 4;
  const px = info.width * info.height;
  let sumAall = 0;
  for (let i = 0; i < px; i += 1) {
    sumAall += data[i * ch + 3] || 0;
  }
  const meanCutoutAlpha = sumAall / Math.max(1, px);
  if (meanCutoutAlpha < 8) {
    console.warn(
      `[switchx:composition] compositeCutoutOnBackground: alpha médio do cutout RGBA=${meanCutoutAlpha.toFixed(2)}/255 — a figura pode desaparecer no vídeo final.`,
    );
  }

  let sumX = 0;
  let sumW = 0;
  for (let y = Math.floor(info.height * 0.6); y < info.height; y += 1) {
    for (let x = 0; x < info.width; x += 1) {
      const alphaIdx = (y * info.width + x) * ch + 3;
      const a = data[alphaIdx] || 0;
      sumX += x * a;
      sumW += a;
    }
  }
  const personCenterX = sumW > 0 ? Math.round(sumX / sumW) : Math.round(ow / 2);

  const shadowSvg = Buffer.from(
    `<svg width="${ow}" height="${oh}" xmlns="http://www.w3.org/2000/svg">
      <ellipse cx="${personCenterX}" cy="${oh - 36}" rx="130" ry="28" fill="rgba(0,0,0,0.38)"/>
    </svg>`,
  );

  const shadow = await sharp(shadowSvg).blur(16).ensureAlpha().png().toBuffer();

  return sharp(bgLayerBuffer)
    .composite([
      { input: shadow, left: 0, top: 0, blend: 'over' },
      { input: cutoutRgba, left: 0, top: 0, blend: 'over' },
    ])
    .png()
    .toBuffer();
}
