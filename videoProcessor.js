/**
 * FFmpeg: extração de frames, informação do vídeo, montagem do MP4.
 */
import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegPath from 'ffmpeg-static';
import ffprobe from 'ffprobe-static';

ffmpeg.setFfmpegPath(ffmpegPath);
ffmpeg.setFfprobePath(ffprobe.path);

/** Ordena só `frame_NNNNN.png` por índice numérico (alinhado com mask_pipeline.run_batch). */
export function sortExtractedFramePaths(absPaths) {
  const re = /^frame_(\d+)\.png$/i;
  return absPaths
    .filter((p) => re.test(path.basename(p)))
    .sort((a, b) => {
      const na = parseInt(path.basename(a).match(re)[1], 10);
      const nb = parseInt(path.basename(b).match(re)[1], 10);
      return na - nb;
    });
}

function parseFpsRate(fracStr) {
  const parts = String(fracStr || '0/1').split('/');
  const a = parseFloat(parts[0]);
  const b = parseFloat(parts[1]);
  if (!Number.isFinite(a) || !Number.isFinite(b) || b <= 0) return null;
  return a / b;
}

/**
 * @param {string} videoPath
 * @returns {Promise<{ duration: number; nativeFps: number; fps: number; hasAudio: boolean; width: number; height: number }>}
 */
export function probeVideo(videoPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(videoPath, (err, metadata) => {
      if (err) {
        reject(err);
        return;
      }
      const dur = parseFloat(metadata.format.duration) || 0;
      const vStream = metadata.streams.find((s) => s.codec_type === 'video');
      let nativeFps =
        parseFpsRate(vStream?.avg_frame_rate) ||
        parseFpsRate(vStream?.r_frame_rate) ||
        30;
      nativeFps = Math.min(120, Math.max(1, nativeFps));
      const fps = Math.min(60, Math.max(8, nativeFps));
      const hasAudio = metadata.streams?.some((s) => s.codec_type === 'audio') ?? false;
      const width = Math.max(1, parseInt(String(vStream?.width ?? 1280), 10) || 1280);
      const height = Math.max(1, parseInt(String(vStream?.height ?? 720), 10) || 720);
      resolve({ duration: dur, nativeFps, fps, hasAudio, width, height });
    });
  });
}

/**
 * Extrai frames PNG 960×540 com amostragem `extractFps`.
 * @param {object} [opts]
 * @param {number} [opts.startSec=0] Início do clip no vídeo (segundos).
 * @param {number} [opts.endSec] Fim exclusivo ou fim do intervalo — usa `endSec - startSec` como duração.
 * @param {number} [opts.maxDuration=5] Se `endSec` omitido, duração a partir de `startSec` (legado).
 * @param {number} [opts.extractFps=8] FPS da sequência extraída (amostragem).
 * @param {number} [opts.outWidth=960] Largura do canvas (par).
 * @param {number} [opts.outHeight=540] Altura do canvas (par).
 */
export async function extractFrames(videoPath, outDir, opts = {}) {
  const extractFps = opts.extractFps ?? 8;
  const startSec = Math.max(0, Number(opts.startSec) || 0);
  let segmentSec;
  if (opts.endSec != null && opts.endSec !== '' && Number.isFinite(Number(opts.endSec))) {
    segmentSec = Math.max(0.04, Number(opts.endSec) - startSec);
  } else {
    segmentSec = Number(opts.maxDuration) > 0 ? Number(opts.maxDuration) : 5;
  }

  await fs.mkdir(outDir, { recursive: true });

  const ow = Math.max(2, Number(opts.outWidth) || 960);
  const oh = Math.max(2, Number(opts.outHeight) || 540);
  const vf = `fps=${extractFps},scale=${ow}:${oh}:force_original_aspect_ratio=decrease,pad=${ow}:${oh}:(ow-iw)/2:(oh-ih)/2:black`;
  const inPath = path.resolve(videoPath).replace(/\\/g, '/');
  const outPattern = path.join(outDir, 'frame_%05d.png').replace(/\\/g, '/');

  await new Promise((resolve, reject) => {
    ffmpeg(inPath)
      .inputOptions(['-ss', String(startSec), '-t', String(segmentSec)])
      .outputOptions(['-vf', vf, '-an'])
      .output(outPattern)
      .on('end', resolve)
      .on('error', reject)
      .run();
  });

  const files = await fs.readdir(outDir);
  const abs = files.filter((f) => f.endsWith('.png')).map((f) => path.join(outDir, f));
  return sortExtractedFramePaths(abs);
}

/**
 * Um único PNG no instante `timeSec` (mesmo scale/pad que extractFrames).
 * @param {{ timeSec?: number; outWidth?: number; outHeight?: number }} opts
 */
export async function extractSingleFrame(videoPath, outPngPath, opts = {}) {
  const timeSec = Math.max(0, Number(opts.timeSec) || 0);
  const ow = Math.max(2, Number(opts.outWidth) || 960);
  const oh = Math.max(2, Number(opts.outHeight) || 540);
  const vf = `scale=${ow}:${oh}:force_original_aspect_ratio=decrease,pad=${ow}:${oh}:(ow-iw)/2:(oh-ih)/2:black`;
  const inPath = path.resolve(videoPath).replace(/\\/g, '/');
  const dest = path.resolve(outPngPath).replace(/\\/g, '/');

  await new Promise((resolve, reject) => {
    ffmpeg(inPath)
      .inputOptions(['-ss', String(timeSec)])
      .outputOptions(['-vframes', '1', '-vf', vf, '-an'])
      .output(dest)
      .on('end', resolve)
      .on('error', reject)
      .run();
  });
}

/**
 * Monta MP4 a partir de PNGs `out_00001.png` (já compostos em RGB no Sharp).
 * O alpha da pessoa é aplicado em `composition.js` (joinChannel + overlay), não aqui com alphamerge.
 */
export async function createVideoFromFrames(framesDir, outputPath, fps) {
  const listed = await fs.readdir(framesDir);
  const outs = listed.filter((f) => /^out_\d+\.png$/i.test(f));
  if (!outs.length) {
    throw new Error('Nenhum frame processado para o vídeo final.');
  }

  const pattern = path.join(framesDir, 'out_%05d.png').replace(/\\/g, '/');
  const outFile = path.resolve(outputPath).replace(/\\/g, '/');
  const bin = ffmpegPath.replace(/\\/g, '/');

  const args = [
    '-hide_banner',
    '-loglevel',
    'error',
    '-y',
    '-framerate',
    String(fps),
    '-start_number',
    '1',
    '-i',
    pattern,
    '-vf',
    'format=rgb24,format=yuv420p',
    '-c:v',
    'libx264',
    '-profile:v',
    'high',
    '-pix_fmt',
    'yuv420p',
    '-preset',
    'veryfast',
    '-crf',
    '18',
    '-movflags',
    '+faststart',
    outFile,
  ];

  const logLine = [bin, ...args]
    .map((a) => (/[\s"]/.test(String(a)) ? `"${String(a).replace(/"/g, '\\"')}"` : String(a)))
    .join(' ');
  console.log(`[switchx:ffmpeg] ${logLine}`);

  await new Promise((resolve, reject) => {
    const p = spawn(bin, args, { windowsHide: true });
    let errBuf = '';
    p.stderr?.on('data', (d) => {
      errBuf += d.toString();
    });
    p.on('error', reject);
    p.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(errBuf.trim() || `ffmpeg terminou com código ${code}`));
    });
  });
}

/**
 * Junta a faixa de áudio do vídeo original ao MP4 já codificado (vídeo sem áudio).
 * Mesma ideia que `referencia/app.py` (ffmpeg -map 0:v -map 1:a?).
 * @param {string} videoOnlyPath MP4 com vídeo H.264 (sem áudio)
 * @param {string} audioSourcePath Vídeo original (ou ficheiro com áudio)
 * @param {string} outputPath Destino final (substitui o conteúdo)
 * @param {{ startSec?: number; durationSec?: number }} [trimAudio] Recorte da faixa de áudio no ficheiro de origem (alinhado ao clip de vídeo)
 */
export async function muxAudioFromSource(videoOnlyPath, audioSourcePath, outputPath, trimAudio = null) {
  const tmpOut = `${outputPath}.muxtmp.mp4`;
  const bin = ffmpegPath.replace(/\\/g, '/');
  const vIn = path.resolve(videoOnlyPath).replace(/\\/g, '/');
  const aIn = path.resolve(audioSourcePath).replace(/\\/g, '/');
  const outTmp = path.resolve(tmpOut).replace(/\\/g, '/');

  const start = trimAudio?.startSec != null ? Math.max(0, Number(trimAudio.startSec) || 0) : null;
  const dur = trimAudio?.durationSec != null ? Math.max(0.05, Number(trimAudio.durationSec) || 0) : null;

  const args = [
    '-hide_banner',
    '-loglevel',
    'error',
    '-y',
    '-i',
    vIn,
  ];
  if (start != null && dur != null) {
    args.push('-ss', String(start), '-t', String(dur));
  }
  args.push(
    '-i',
    aIn,
    '-map',
    '0:v:0',
    '-map',
    '1:a?',
    '-c:v',
    'copy',
    '-c:a',
    'aac',
    '-b:a',
    '160k',
    '-shortest',
    outTmp,
  );

  await new Promise((resolve, reject) => {
    const p = spawn(bin, args, { windowsHide: true });
    let errBuf = '';
    p.stderr?.on('data', (d) => {
      errBuf += d.toString();
    });
    p.on('error', reject);
    p.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(errBuf.trim() || `ffmpeg mux terminou com código ${code}`));
    });
  });

  await fs.unlink(outputPath).catch(() => {});
  await fs.rename(tmpOut, outputPath);
}

export async function safeRm(dir) {
  try {
    await fs.rm(dir, { recursive: true, force: true });
  } catch {
    /* ignore */
  }
}
