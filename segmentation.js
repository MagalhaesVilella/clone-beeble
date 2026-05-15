/**
 * Segmentação: batch via worker Python ou caminhos rápidos (fill / upload) em Sharp.
 */
import { spawn } from 'child_process';
import { randomUUID } from 'crypto';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import sharp from 'sharp';
import { getPythonBin, isPythonWorkerReady } from './pythonHealth.js';
import { sortExtractedFramePaths } from './videoProcessor.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
/** CWD dos workers Python: imports relativos e paths estáveis (evita falhas silenciosas no batch). */
const PY_CWD = path.join(__dirname, 'python');
const PY_SCRIPT = path.join(PY_CWD, 'mask_pipeline.py');
const PY_PIPELINE_SCRIPT = path.join(PY_CWD, 'pipeline.py');
const PY_RVM_SCRIPT = path.join(PY_CWD, 'rvm_video_mask.py');
const PY_SAM2_SCRIPT = path.join(PY_CWD, 'sam2_video_mask.py');

/** RVM/SAM2 só correm se isto E o flag específico estiverem activos (MVP estável = só BiRefNet+MODNet em batch). */
function allowExperimentalVideoMotors() {
  return process.env.SWITCHX_ALLOW_RVM_SAM2 === '1' || String(process.env.SWITCHX_ALLOW_RVM_SAM2 || '').toLowerCase() === 'true';
}

function useRvmVideo() {
  const v = process.env.SWITCHX_USE_RVM;
  return v === '1' || String(v || '').toLowerCase() === 'true';
}

function useSam2Video() {
  const v = process.env.SWITCHX_USE_SAM2;
  return v === '1' || String(v || '').toLowerCase() === 'true';
}

function pythonCmd() {
  return getPythonBin();
}

function logMaskErr(chunk) {
  const s = chunk.toString();
  if (!s) return;
  for (const line of s.split('\n')) {
    if (line.trim()) console.error('[switchx:mask]', line);
  }
}

function runPythonScript(scriptPath, scriptArgs) {
  return new Promise((resolve, reject) => {
    const p = spawn(pythonCmd(), [scriptPath, ...scriptArgs], {
      windowsHide: true,
      cwd: PY_CWD,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });
    let stderr = '';
    p.stderr?.on('data', (d) => {
      stderr += d.toString();
      logMaskErr(d);
    });
    p.on('error', reject);
    p.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(stderr.trim() || `script Python terminou com código ${code}`));
    });
  });
}

async function readMaskReport(reportPath) {
  const reportRaw = await fs.readFile(reportPath, 'utf-8').catch(() => '{}');
  try {
    return JSON.parse(String(reportRaw).trim() || '{}');
  } catch {
    console.warn('[switchx:mask] report.json inválido ou vazio; a assumir {}');
    return {};
  }
}

/** Média 0–255 de buffer grayscale raw. */
function meanGrayU8(buf) {
  if (!buf?.length) return 0;
  let s = 0;
  for (let i = 0; i < buf.length; i += 1) s += buf[i];
  return s / buf.length;
}

async function meanPngGrayscale(pngPath) {
  try {
    const { data } = await sharp(pngPath).grayscale().raw().toBuffer({ resolveWithObject: true });
    return meanGrayU8(data);
  } catch {
    return 0;
  }
}

/** Máscaras RVM/SAM2 quase pretas (erro silencioso / modelo falhou). */
const ALPHA_BATCH_SUSPICIOUS_MAX = 12; // /255
async function batchMasksLookEmpty(maskDir, expectedCount) {
  const first = path.join(maskDir, 'mask_00001.png');
  try {
    await fs.access(first);
  } catch {
    return true;
  }
  const m1 = await meanPngGrayscale(first);
  if (m1 < ALPHA_BATCH_SUSPICIOUS_MAX) return true;
  const mid = Math.min(Math.max(1, expectedCount), Math.max(1, Math.ceil(expectedCount / 2)));
  const midPath = path.join(maskDir, `mask_${String(mid).padStart(5, '0')}.png`);
  const m2 = await meanPngGrayscale(midPath).catch(() => 0);
  if (m2 < ALPHA_BATCH_SUSPICIOUS_MAX) return true;
  return false;
}

function runPythonMask(args) {
  return new Promise((resolve, reject) => {
    const p = spawn(pythonCmd(), [PY_SCRIPT, ...args], {
      windowsHide: true,
      cwd: PY_CWD,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });
    let stderr = '';
    let stdout = '';
    p.stdout?.on('data', (d) => {
      stdout += d.toString();
    });
    p.stderr?.on('data', (d) => {
      stderr += d.toString();
      logMaskErr(d);
    });
    p.on('error', reject);
    p.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(stderr.trim() || stdout.trim() || `mask worker terminou com código ${code}`));
    });
  });
}

async function alphaFromUserMaskFile(userMaskPath, w, h) {
  const raw = await sharp(userMaskPath)
    .rotate()
    .resize(w, h, { fit: 'fill', position: 'centre' })
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return raw.data;
}

/**
 * @param {string} imagePath - PNG do frame (960×540)
 * @param {{
 *   mode?: 'auto'|'select'|'fill'|'upload';
 *   points?: Array<{x:number;y:number}>;
 *   userMaskPath?: string;
 *   previewPath?: string;
 * }} opts
 * @returns {Promise<{ width: number; height: number; alpha: Buffer }>}
 */
export async function segmentPersonAlpha(imagePath, opts = {}) {
  const meta = await sharp(imagePath).metadata();
  const w = meta.width;
  const h = meta.height;
  const mode = opts.mode || 'auto';

  if (mode === 'fill') {
    const alpha = Buffer.alloc(w * h, 255);
    if (opts.previewPath) {
      await sharp(Buffer.alloc(w * h, 255), { raw: { width: w, height: h, channels: 1 } })
        .png()
        .toFile(opts.previewPath)
        .catch(() => {});
    }
    return { width: w, height: h, alpha };
  }

  if (mode === 'upload' && opts.userMaskPath) {
    const alpha = await alphaFromUserMaskFile(opts.userMaskPath, w, h);
    if (opts.previewPath) {
      const bin = await sharp(alpha, { raw: { width: w, height: h, channels: 1 } })
        .threshold(128)
        .png()
        .toBuffer();
      await fs.writeFile(opts.previewPath, bin).catch(() => {});
    }
    return { width: w, height: h, alpha };
  }

  const tmpMaskPath = `${imagePath}.alpha.png`;
  const args = ['--input', imagePath, '--output', tmpMaskPath, '--mode', mode];
  if (opts.points?.length) {
    args.push('--points', JSON.stringify(opts.points));
  }
  if (opts.userMaskPath) {
    args.push('--user-mask', opts.userMaskPath);
  }
  if (opts.previewPath) {
    args.push('--preview', opts.previewPath);
  }
  await runPythonMask(args);
  const alphaPng = await fs.readFile(tmpMaskPath);
  const alpha = await sharp(alphaPng).resize(w, h, { kernel: sharp.kernel.cubic }).raw().toBuffer();
  await fs.unlink(tmpMaskPath).catch(() => {});
  return { width: w, height: h, alpha };
}

/**
 * @param {string[]} imagePaths
 * @param {{
 *   mode?: 'auto'|'select'|'fill'|'upload';
 *   points?: Array<{x:number;y:number}>;
 *   userMaskPath?: string;
 *   previewPath?: string;
 *   promptText?: string;
 * }} opts
 */
export async function segmentFramesAlpha(imagePaths, opts = {}) {
  if (!imagePaths?.length) return [];

  const sorted = sortExtractedFramePaths(imagePaths);
  const refFramePath = sorted[0];
  console.log(
    `[switchx:mask] Frame de referência (1.º do clipe extraído, ordem temporal): ${path.basename(refFramePath)} → ${refFramePath}`,
  );
  const mode = opts.mode || 'auto';

  let effectiveMode = mode;
  let pointsForPy = opts.points?.length ? opts.points : [];
  if (mode === 'select' && !pointsForPy.length) {
    console.warn('[switchx:mask] modo select sem pontos — fallback para auto');
    effectiveMode = 'auto';
    pointsForPy = [];
  }

  if (mode === 'fill') {
    const out = [];
    for (let i = 0; i < sorted.length; i += 1) {
      const imagePath = sorted[i];
      const meta = await sharp(imagePath).metadata();
      const w = meta.width;
      const h = meta.height;
      const alpha = Buffer.alloc(w * h, 255);
      if (i === 0 && opts.previewPath) {
        const bin = Buffer.alloc(w * h, 255);
        await sharp(bin, { raw: { width: w, height: h, channels: 1 } })
          .png()
          .toFile(opts.previewPath)
          .catch(() => {});
      }
      out.push({ width: w, height: h, alpha });
    }
    return out;
  }

  if (!isPythonWorkerReady()) {
    throw new Error(
      'Worker Python indisponível: executa `pip install -r requirements.txt` no mesmo Python que PYTHON_BIN e reinicia o servidor.',
    );
  }

  const inputDirAbs = path.resolve(path.dirname(sorted[0]));
  const tmpDir = path.resolve(await fs.mkdtemp(path.join(os.tmpdir(), 'mask-batch-')));
  const reportPath = path.join(os.tmpdir(), `swx-mask-report-${randomUUID()}.json`);

  const pipelineArgs = () => {
    const a = [
      '--frames-dir',
      inputDirAbs,
      '--masks-dir',
      tmpDir,
      '--mode',
      effectiveMode,
      '--report',
      reportPath,
    ];
    if (pointsForPy.length) a.push('--points', JSON.stringify(pointsForPy));
    if (opts.userMaskPath) a.push('--user-mask', opts.userMaskPath);
    if (opts.previewPath) a.push('--preview', opts.previewPath);
    return a;
  };

  const videoMotorCandidate =
    (effectiveMode === 'auto' || effectiveMode === 'select') && !opts.userMaskPath;

  let report = {};
  let usedRvm = false;
  let usedSam2 = false;
  let repairedFromEmptyVideoMotor = false;

  if (videoMotorCandidate && useRvmVideo() && !allowExperimentalVideoMotors()) {
    console.warn(
      '[switchx:mask] SWITCHX_USE_RVM está activo mas RVM está bloqueado: define também SWITCHX_ALLOW_RVM_SAM2=1 (senão só BiRefNet+MODNet).',
    );
  }
  if (videoMotorCandidate && useSam2Video() && !allowExperimentalVideoMotors()) {
    console.warn(
      '[switchx:mask] SWITCHX_USE_SAM2 está activo mas SAM2 está bloqueado: define também SWITCHX_ALLOW_RVM_SAM2=1 (senão só BiRefNet+MODNet).',
    );
  }

  if (allowExperimentalVideoMotors() && videoMotorCandidate && useRvmVideo()) {
    const rvmArgs = [
      '--frames-dir',
      inputDirAbs,
      '--masks-dir',
      tmpDir,
      '--mode',
      effectiveMode,
      '--report',
      reportPath,
    ];
    if (pointsForPy.length) rvmArgs.push('--points', JSON.stringify(pointsForPy));
    if (opts.previewPath) rvmArgs.push('--preview', opts.previewPath);
    if (opts.promptText) rvmArgs.push('--prompt-text', opts.promptText);
    try {
      await runPythonScript(PY_RVM_SCRIPT, rvmArgs);
      report = await readMaskReport(reportPath);
      if (report.rvm_ok) {
        usedRvm = true;
        console.log('[switchx:mask] motor vídeo: RVM (batch concluído, a validar máscaras…)');
      } else {
        console.warn(
          '[switchx:mask] RVM não disponível ou falhou —',
          report.error || 'rvm_ok=false',
          'a tentar SAM2 ou pipeline BiRefNet.',
        );
      }
    } catch (e) {
      console.warn('[switchx:mask] worker RVM falhou — a tentar SAM2 ou pipeline:', e?.message || e);
    }
  }

  if (allowExperimentalVideoMotors() && !usedRvm && videoMotorCandidate && useSam2Video()) {
    const sam2Args = [
      '--frames-dir',
      inputDirAbs,
      '--masks-dir',
      tmpDir,
      '--mode',
      effectiveMode,
      '--prompt-text',
      opts.promptText || 'person',
      '--report',
      reportPath,
    ];
    if (pointsForPy.length) sam2Args.push('--points', JSON.stringify(pointsForPy));
    if (opts.previewPath) sam2Args.push('--preview', opts.previewPath);
    try {
      await runPythonScript(PY_SAM2_SCRIPT, sam2Args);
      report = await readMaskReport(reportPath);
      if (report.sam2_ok) {
        usedSam2 = true;
        console.log('[switchx:mask] motor vídeo: SAM2 (batch concluído, a validar máscaras…)');
      } else {
        console.warn(
          '[switchx:mask] SAM 2.1 não disponível ou falhou —',
          report.error || 'sam2_ok=false',
          'a usar pipeline BiRefNet.',
        );
      }
    } catch (e) {
      console.warn('[switchx:mask] worker SAM2 falhou — a usar pipeline BiRefNet:', e?.message || e);
    }
  }

  if (!usedRvm && !usedSam2) {
    console.log('[switchx:mask] motor: pipeline BiRefNet + MODNet (+ rembg) em batch');
    await runPythonScript(PY_PIPELINE_SCRIPT, pipelineArgs());
    report = await readMaskReport(reportPath);
  }

  const maskDir = path.join(tmpDir, '_batch_masks');
  if ((usedRvm || usedSam2) && (await batchMasksLookEmpty(maskDir, sorted.length))) {
    console.warn(
      '[switchx:mask] ALERTA: máscaras do motor vídeo (RVM/SAM2) quase vazias ou inválidas — a repor com pipeline BiRefNet+MODNet em batch.',
    );
    usedRvm = false;
    usedSam2 = false;
    repairedFromEmptyVideoMotor = true;
    await runPythonScript(PY_PIPELINE_SCRIPT, pipelineArgs());
    report = await readMaskReport(reportPath);
  }

  const requiresAiMask = effectiveMode === 'auto' || effectiveMode === 'select';
  const okMasks =
    (usedRvm && report.rvm_ok) ||
    (usedSam2 && report.sam2_ok) ||
    report.birefnet_ok ||
    report.modnet_ok ||
    report.rembg_ok;
  if (requiresAiMask && !okMasks) {
    throw new Error(
      'Segmentação indisponível: RVM/SAM2 (se ativos), BiRefNet, MODNet e rembg falharam. Verifica os logs [switchx:mask] e dependências Python.',
    );
  }

  let trustBatchPngs = true;
  if (requiresAiMask && (await batchMasksLookEmpty(maskDir, sorted.length))) {
    trustBatchPngs = false;
    console.warn(
      '[switchx:mask] CRÍTICO: PNGs do batch parecem vazios (média < limiar) — a usar BiRefNet+MODNet frame-a-frame por cada frame (mesmo código que o preview).',
    );
  }

  const out = [];
  let goodMasks = 0;
  let perFrameRescue = 0;
  let fillLastResort = 0;
  for (let i = 0; i < sorted.length; i += 1) {
    const imagePath = sorted[i];
    const meta = await sharp(imagePath).metadata();
    const w = meta.width;
    const h = meta.height;
    let alpha;
    const name = `mask_${String(i + 1).padStart(5, '0')}.png`;
    const filePreferred = path.join(maskDir, name);
    const fileLegacy = path.join(tmpDir, name);
    let alphaPng = null;
    if (trustBatchPngs) {
      alphaPng = await fs.readFile(filePreferred).catch(() => null);
      if (!alphaPng) alphaPng = await fs.readFile(fileLegacy).catch(() => null);
    }
    if (!alphaPng) {
      console.warn(
        `[switchx:mask] PNG em falta (${name}); fallback frame-a-frame (BiRefNet) para ${path.basename(imagePath)}`,
      );
      try {
        const one = await segmentPersonAlpha(imagePath, {
          mode: effectiveMode,
          points: pointsForPy,
          userMaskPath: opts.userMaskPath,
          previewPath: undefined,
        });
        alpha = one.alpha;
        perFrameRescue += 1;
      } catch (fe) {
        console.error(`[switchx:mask] fallback frame-a-frame falhou frame ${i + 1}:`, fe?.message || fe);
        console.warn('[switchx:mask] último recurso: alpha=255 (fill opaco) neste frame.');
        alpha = Buffer.alloc(w * h, 255);
        fillLastResort += 1;
      }
    } else {
      alpha = await sharp(alphaPng).resize(w, h, { kernel: sharp.kernel.cubic }).raw().toBuffer();
    }

    if (requiresAiMask) {
      const m = meanGrayU8(alpha);
      if (m < 12) {
        console.warn(
          `[switchx:mask] frame ${i + 1}/${sorted.length} alpha médio muito baixo (${m.toFixed(1)}/255) — retry BiRefNet (1 frame)`,
        );
        try {
          const one = await segmentPersonAlpha(imagePath, {
            mode: effectiveMode,
            points: pointsForPy,
            userMaskPath: opts.userMaskPath,
            previewPath: undefined,
          });
          alpha = one.alpha;
          perFrameRescue += 1;
        } catch (fe) {
          console.error(`[switchx:mask] retry falhou frame ${i + 1}:`, fe?.message || fe);
          alpha = Buffer.alloc(w * h, 255);
          fillLastResort += 1;
        }
      }
      const m2 = meanGrayU8(alpha);
      if (m2 < 12) {
        console.warn(
          `[switchx:mask] frame ${i + 1} continua quase vazio (média ${m2.toFixed(1)}) — fill opaco (último recurso).`,
        );
        alpha = Buffer.alloc(w * h, 255);
        fillLastResort += 1;
      }
    }

    const mFinal = meanGrayU8(alpha);
    if (requiresAiMask && mFinal >= 12) goodMasks += 1;

    out.push({ width: w, height: h, alpha });
  }

  const firstMean = out.length ? meanGrayU8(out[0].alpha) : 0;
  if (requiresAiMask && firstMean < 255 * 0.05) {
    console.warn(
      `[switchx:mask] ALERTA FORTE: alpha médio no 1.º frame após correções = ${firstMean.toFixed(1)}/255 (< 5%). Verifica o clip e o motor.`,
    );
  }

  let motorLabel = 'BiRefNet+MODNet(batch)';
  if (repairedFromEmptyVideoMotor) motorLabel = 'BiRefNet+MODNet(batch) [substituiu RVM/SAM2 vazio]';
  else if (usedRvm) motorLabel = 'RVM';
  else if (usedSam2) motorLabel = 'SAM2';
  if (!trustBatchPngs) {
    motorLabel = `${motorLabel} → per-frame (batch PNGs inválidos)`;
  }
  console.log(
    `[switchx:mask] resumo: motor=${motorLabel} · frames=${sorted.length} · máscaras “boas”≈${goodMasks} · rescues frame=${perFrameRescue} · fill último recurso=${fillLastResort}`,
  );
  await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  await fs.unlink(reportPath).catch(() => {});

  const byOriginalOrder = [];
  const norm = (p) => path.resolve(p);
  const indexByPath = new Map(sorted.map((p, idx) => [norm(p), idx]));
  for (const p of imagePaths) {
    const idx = indexByPath.get(norm(p));
    if (idx !== undefined) byOriginalOrder.push(out[idx]);
  }
  return byOriginalOrder.length === imagePaths.length ? byOriginalOrder : out;
}

export { isPythonWorkerReady } from './pythonHealth.js';
