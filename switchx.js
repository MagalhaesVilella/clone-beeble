import express from 'express';
import fs from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { randomUUID } from 'crypto';
import sharp from 'sharp';
import { isPythonWorkerReady, getPythonBin } from '../services/pythonHealth.js';
import { segmentFramesAlpha, segmentPersonAlpha } from '../services/segmentation.js';
import { analyzeReferenceLighting, computeGlobalRelight, applyGlobalRelight } from '../services/relighting.js';
import { makeCutout, getBackgroundLayer, compositeCutoutOnBackground } from '../services/composition.js';
import {
  probeVideo,
  extractFrames,
  extractSingleFrame,
  createVideoFromFrames,
  muxAudioFromSource,
  safeRm,
} from '../services/videoProcessor.js';
import { computeOutputFrameSize } from '../services/outputDimensions.js';
import { ensureSceneDirs, writeSceneManifest } from '../services/sceneSession.js';
import { mergeRelightPreserveHighFreq } from '../services/preserveRelight.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const PY_CWD = path.join(ROOT, 'services', 'python');
const RELIGHT_LAP_SCRIPT = path.join(PY_CWD, 'relight_laplace.py');
const MAX_NATIVE_FRAMES = 240;
const MAX_SOURCE_DURATION_SEC = 600;

/** Pirâmide Laplaciana (OpenCV) entre PNGs RGB pré/pós re-light. */
function laplacianMergePngPaths(prePath, postPath, outPath) {
  return new Promise((resolve, reject) => {
    const bin = getPythonBin();
    const args = [RELIGHT_LAP_SCRIPT, 'merge', '--before', prePath, '--after', postPath, '--out', outPath];
    const p = spawn(bin, args, { windowsHide: true, cwd: PY_CWD });
    let err = '';
    p.stderr?.on('data', (d) => {
      err += d.toString();
    });
    p.on('error', reject);
    p.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(err.trim() || `relight_laplace.py terminou com código ${code}`));
    });
  });
}

export function createSwitchxRouter({ upload }) {
  const router = express.Router();

  router.post('/probe', upload.single('video'), async (req, res) => {
    if (!req.file?.path) {
      res.status(400).json({ error: 'Envie o vídeo no campo "video".' });
      return;
    }
    const p = req.file.path;
    try {
      const meta = await probeVideo(p);
      const maxSeg = Math.min(meta.duration, MAX_NATIVE_FRAMES / Math.max(1, meta.nativeFps));
      await fs.unlink(p).catch(() => {});
      res.json({
        ok: true,
        duration: meta.duration,
        nativeFps: meta.nativeFps,
        fps: meta.fps,
        hasAudio: meta.hasAudio,
        width: meta.width,
        height: meta.height,
        maxNativeFrames: MAX_NATIVE_FRAMES,
        maxSegmentSec: maxSeg,
      });
    } catch (err) {
      await fs.unlink(p).catch(() => {});
      res.status(400).json({ error: err.message || 'Falha ao analisar o vídeo.' });
    }
  });

  router.post(
    '/preview-mask',
    upload.fields([
      { name: 'video', maxCount: 1 },
      { name: 'maskUpload', maxCount: 1 },
    ]),
    async (req, res) => {
      const jobId = randomUUID();
      const jobDir = path.join(ROOT, 'tmp', `pm-${jobId}`);
      const videoPath = req.files?.video?.[0]?.path;
      const userMaskUpload = req.files?.maskUpload?.[0]?.path || null;
      try {
        if (!videoPath) {
          res.status(400).json({ error: 'Envie o vídeo (campo video).' });
          return;
        }
        const ALLOWED_MASK = new Set(['auto', 'select', 'fill', 'upload']);
        let maskMode = String(req.body?.maskMode || 'auto').toLowerCase();
        if (!ALLOWED_MASK.has(maskMode)) maskMode = 'auto';
        let selectedPoints = [];
        const rawPoints = req.body?.maskPoints;
        if (rawPoints) {
          try {
            const parsed = typeof rawPoints === 'string' ? JSON.parse(rawPoints) : rawPoints;
            selectedPoints = Array.isArray(parsed) ? parsed : [];
          } catch {
            selectedPoints = [];
          }
        }
        if (maskMode === 'upload' && !userMaskUpload) {
          await fs.unlink(videoPath).catch(() => {});
          res.status(400).json({ error: 'Modo máscara "upload" requer maskUpload.' });
          return;
        }
        if (maskMode !== 'fill' && !isPythonWorkerReady()) {
          await fs.unlink(videoPath).catch(() => {});
          res.status(503).json({
            error:
              'Worker Python indisponível. Instala dependências em backend e reinicia a API (PYTHON_BIN).',
          });
          return;
        }
        const outputResolution = String(req.body?.outputResolution || '1080p').toLowerCase();
        const { duration, width: srcW, height: srcH } = await probeVideo(videoPath);
        let clipStart = Math.max(0, parseFloat(String(req.body?.clipStartSec ?? '0')) || 0);
        clipStart = Math.min(clipStart, Math.max(0, duration - 0.04));
        const outDims = computeOutputFrameSize(srcW, srcH, outputResolution);

        await fs.mkdir(jobDir, { recursive: true });
        const framePng = path.join(jobDir, 'preview_frame.png');
        await extractSingleFrame(videoPath, framePng, {
          timeSec: clipStart,
          outWidth: outDims.width,
          outHeight: outDims.height,
        });

        let userMaskPathForSeg = null;
        if (maskMode === 'upload' && userMaskUpload) {
          userMaskPathForSeg = path.join(jobDir, 'user_mask.png');
          await sharp(userMaskUpload)
            .rotate()
            .resize(outDims.width, outDims.height, { fit: 'fill', position: 'centre' })
            .grayscale()
            .png()
            .toFile(userMaskPathForSeg);
        }

        const mrefDir = path.join(jobDir, 'mask_ref');
        await fs.mkdir(mrefDir, { recursive: true });
        const refNamed = path.join(mrefDir, 'frame_00001.png');
        await fs.copyFile(framePng, refNamed);

        const previewFile = `mask_preview_${jobId}.png`;
        const previewPath = path.join(ROOT, 'outputs', previewFile);
        const maskPromptText = String(req.body?.maskPromptText || req.body?.maskPrompt || '').trim();

        console.log('[switchx] preview-mask: segmentação com o mesmo fluxo que o batch (segmentFramesAlpha, 1 frame).');
        await segmentFramesAlpha([refNamed], {
          mode: maskMode,
          points: selectedPoints,
          userMaskPath: userMaskPathForSeg,
          previewPath,
          promptText: maskPromptText || undefined,
        });

        await safeRm(jobDir);
        await fs.unlink(videoPath).catch(() => {});
        if (userMaskUpload) await fs.unlink(userMaskUpload).catch(() => {});

        res.json({
          ok: true,
          maskPreviewUrl: `/outputs/${previewFile}`,
          outWidth: outDims.width,
          outHeight: outDims.height,
        });
      } catch (err) {
        console.error('[switchx:preview-mask]', err);
        await safeRm(jobDir).catch(() => {});
        await fs.unlink(videoPath).catch(() => {});
        if (userMaskUpload) await fs.unlink(userMaskUpload).catch(() => {});
        res.status(500).json({ error: err.message || 'Falha na pré-visualização da máscara.' });
      }
    },
  );

  router.post(
    '/',
    upload.fields([
      { name: 'video', maxCount: 1 },
      { name: 'background', maxCount: 1 },
      { name: 'reference', maxCount: 1 },
      { name: 'maskUpload', maxCount: 1 },
    ]),
    async (req, res) => {
    const jobId = randomUUID();
    const jobDir = path.join(ROOT, 'tmp', jobId);
    const originalDir = path.join(jobDir, 'original');
    const processedDir = path.join(jobDir, 'processed');

    try {
      if (!req.files?.video?.[0] || !req.files?.background?.[0]) {
        res.status(400).json({ error: 'Envie vídeo (video) e imagem de fundo (background).' });
        return;
      }

      const videoPath = req.files.video[0].path;
      const bgPath = req.files.background[0].path;
      const referencePath = req.files?.reference?.[0]?.path || bgPath;
      const userMaskPath = req.files?.maskUpload?.[0]?.path || null;

      const ALLOWED_MASK = new Set(['auto', 'select', 'fill', 'upload']);
      let maskMode = String(req.body?.maskMode || 'auto').toLowerCase();
      if (!ALLOWED_MASK.has(maskMode)) maskMode = 'auto';

      let selectedPoints = [];
      const rawPoints = req.body?.maskPoints;
      if (rawPoints) {
        try {
          const parsed = typeof rawPoints === 'string' ? JSON.parse(rawPoints) : rawPoints;
          selectedPoints = Array.isArray(parsed) ? parsed : [];
        } catch {
          selectedPoints = [];
        }
      }

      const maskPromptText = String(req.body?.maskPromptText || req.body?.maskPrompt || '').trim();

      if (maskMode === 'upload' && !userMaskPath) {
        await fs.unlink(videoPath).catch(() => {});
        await fs.unlink(bgPath).catch(() => {});
        if (referencePath && referencePath !== bgPath) await fs.unlink(referencePath).catch(() => {});
        res.status(400).json({ error: 'Modo máscara "upload" requer um ficheiro (campo maskUpload).' });
        return;
      }

      const model = String(req.body?.model || 'google').toLowerCase();
      if (model !== 'google') {
        res.status(400).json({ error: 'Apenas o modelo google está ativo nesta versão.' });
        return;
      }
      const outputResolution = String(req.body?.outputResolution || '1080p').toLowerCase();
      const quality = String(req.body?.quality || req.body?.maskQuality || 'standard').toLowerCase();
      const extractFps = quality === 'fast' ? 1 : quality === 'precise' ? 4 : 2;
      console.log(
        `[switchx] start job=${jobId} model=${model} quality=${quality} fps=${extractFps} outRes=${outputResolution}`,
      );

      const cleanupUploads = async () => {
        await fs.unlink(videoPath).catch(() => {});
        await fs.unlink(bgPath).catch(() => {});
        if (referencePath && referencePath !== bgPath) await fs.unlink(referencePath).catch(() => {});
        if (userMaskPath) await fs.unlink(userMaskPath).catch(() => {});
      };

      if (maskMode !== 'fill' && !isPythonWorkerReady()) {
        await cleanupUploads();
        res.status(503).json({
          error:
            'Worker Python indisponível. Na pasta backend executa: pip install -r requirements.txt ' +
            '(usa o mesmo interpretador que PYTHON_BIN no .env) e reinicia a API.',
        });
        return;
      }

      const { duration, nativeFps, hasAudio, width: srcW, height: srcH } = await probeVideo(videoPath);
      const outDims = computeOutputFrameSize(srcW, srcH, outputResolution);
      if (duration > MAX_SOURCE_DURATION_SEC) {
        await cleanupUploads();
        res.status(400).json({
          error: `Vídeo demasiado longo (máx. ${MAX_SOURCE_DURATION_SEC}s). Escolhe um ficheiro mais curto.`,
        });
        return;
      }
      if (duration < 0.05) {
        await cleanupUploads();
        res.status(400).json({ error: 'Vídeo demasiado curto.' });
        return;
      }

      let clipStart = Math.max(0, parseFloat(String(req.body?.clipStartSec ?? '0')) || 0);
      let clipEnd = parseFloat(String(req.body?.clipEndSec ?? ''));
      if (!Number.isFinite(clipEnd)) {
        clipEnd = Math.min(duration, clipStart + Math.min(5, MAX_NATIVE_FRAMES / Math.max(1, nativeFps)));
      }
      clipEnd = Math.min(duration, clipEnd);
      clipStart = Math.min(clipStart, Math.max(0, clipEnd - 0.04));
      const clipDur = clipEnd - clipStart;
      const approxNativeFrames = Math.ceil(clipDur * Math.max(1, nativeFps));
      if (clipDur < 0.04) {
        await cleanupUploads();
        res.status(400).json({ error: 'Intervalo inválido: o fim deve ser maior que o início.' });
        return;
      }
      if (clipStart < 0 || clipEnd > duration + 0.01) {
        await cleanupUploads();
        res.status(400).json({ error: 'Intervalo fora da duração do vídeo.' });
        return;
      }
      if (approxNativeFrames > MAX_NATIVE_FRAMES) {
        await cleanupUploads();
        res.status(400).json({
          error: `O trecho tem ~${approxNativeFrames} frames nativos; o máximo é ${MAX_NATIVE_FRAMES}. Encurta o intervalo.`,
        });
        return;
      }

      await fs.mkdir(jobDir, { recursive: true });
      await fs.mkdir(processedDir, { recursive: true });
      await ensureSceneDirs(jobDir);
      const hfPreserve =
        String(req.body?.hfPreserve ?? req.body?.preserveHighFreq ?? 'true').toLowerCase() !== 'false';
      const hfSigma = Math.min(6, Math.max(0.6, parseFloat(String(req.body?.hfSigma ?? '2.2')) || 2.2));
      const hfDetail = Math.min(1.25, Math.max(0, parseFloat(String(req.body?.hfDetail ?? '0.92')) || 0.92));
      console.log(
        `[switchx] job=${jobId} maskMode=${maskMode} canvas=${outDims.width}x${outDims.height}px clip=${clipStart.toFixed(2)}–${clipEnd.toFixed(2)}s hfPreserve=${hfPreserve}`,
      );

      let userMaskPathForSeg = null;
      if (maskMode === 'upload' && userMaskPath) {
        try {
          await fs.access(userMaskPath);
        } catch {
          await safeRm(jobDir);
          await cleanupUploads();
          res.status(400).json({ error: 'Ficheiro de máscara (maskUpload) não encontrado no disco.' });
          return;
        }
        userMaskPathForSeg = path.join(jobDir, 'user_mask_scaled.png');
        await sharp(userMaskPath)
          .rotate()
          .resize(outDims.width, outDims.height, { fit: 'fill', position: 'centre' })
          .grayscale()
          .png()
          .toFile(userMaskPathForSeg);
      }

      console.log(
        `[switchx] extracting frames clip=${clipStart.toFixed(3)}…${clipEnd.toFixed(3)}s (~${approxNativeFrames} @${nativeFps.toFixed(2)}fps nat) extractFps=${extractFps}`,
      );
      const frames = await extractFrames(videoPath, originalDir, {
        startSec: clipStart,
        endSec: clipEnd,
        extractFps,
        outWidth: outDims.width,
        outHeight: outDims.height,
      });
      console.log(`[switchx] extração concluída: ${frames.length} frames em ${originalDir}`);

      await writeSceneManifest({
        jobDir,
        jobId,
        framePaths: frames,
        clipStartSec: clipStart,
        clipEndSec: clipEnd,
        extractFps,
        nativeFps,
        outDims,
        videoProbe: { duration, nativeFps, hasAudio, width: srcW, height: srcH },
      }).catch((e) => console.warn('[switchx] manifest cena ignorado:', e?.message || e));

      if (!frames.length) {
        await safeRm(jobDir);
        await cleanupUploads();
        res.status(500).json({ error: 'Não foi possível extrair frames do vídeo.' });
        return;
      }

      const bgLayer = await getBackgroundLayer(bgPath, outDims);
      const referenceLayer = await getBackgroundLayer(referencePath, outDims);
      const referenceLight = await analyzeReferenceLighting(referenceLayer);

      let relightParams = null;
      let previewMaskUrl = '';
      let alphaFrames;
      try {
        console.log('[switchx] segmenting frames...');
        alphaFrames = await segmentFramesAlpha(frames, {
          mode: maskMode,
          points: selectedPoints,
          userMaskPath: userMaskPathForSeg,
          previewPath: path.join(processedDir, 'mask_preview.png'),
          promptText: maskPromptText || undefined,
        });
        console.log(`[switchx] segmentação concluída: ${alphaFrames.length} alphas`);
      } catch (batchErr) {
        console.warn('[switchx] segmentação batch falhou — fallback frame-a-frame:', batchErr?.message || batchErr);
        alphaFrames = [];
        for (let i = 0; i < frames.length; i += 1) {
          const framePath = frames[i];
          try {
            const one = await segmentPersonAlpha(framePath, {
              mode: maskMode,
              points: selectedPoints,
              userMaskPath: userMaskPathForSeg,
              previewPath: i === 0 ? path.join(processedDir, 'mask_preview.png') : undefined,
            });
            alphaFrames.push(one);
          } catch (fe) {
            console.warn(`[switchx] fallback frame ${i + 1}/${frames.length} falhou:`, fe?.message || fe);
            const meta = await sharp(framePath).metadata();
            const w = meta.width || 960;
            const h = meta.height || 540;
            alphaFrames.push({ width: w, height: h, alpha: Buffer.alloc(w * h, 255) });
          }
        }
        console.log(`[switchx] fallback frame-a-frame terminou: ${alphaFrames.length} alphas`);
      }

      console.log('[switchx] composição (makeCutout + relight + fundo)…');
      let emaBright = null;
      let emaSat = null;
      const relightEma = Math.min(1, Math.max(0.05, parseFloat(String(process.env.SWITCHX_RELIGHT_EMA || '0.32')) || 0.32));
      const usePyLap =
        String(process.env.SWITCHX_LAPLACE_PYTHON || '').toLowerCase() === '1' ||
        String(process.env.SWITCHX_LAPLACE_PYTHON || '').toLowerCase() === 'true';

      for (let i = 0; i < frames.length; i += 1) {
        const framePath = frames[i];
        const currentAlpha = alphaFrames[i];
        if (!currentAlpha) {
          throw new Error(`Falha de máscara no frame ${i + 1}/${frames.length}.`);
        }
        const { alpha, width, height } = currentAlpha;
        if (i === 0) previewMaskUrl = path.join(processedDir, 'mask_preview.png');
        let cutout = await makeCutout(framePath, alpha, width, height, outDims);

        const frameStats = await computeGlobalRelight(cutout, bgLayer);
        const mergedStats = { ...frameStats, ...referenceLight };
        if (emaBright === null) {
          emaBright = mergedStats.brightness;
          emaSat = mergedStats.saturation;
        } else {
          emaBright = relightEma * mergedStats.brightness + (1 - relightEma) * emaBright;
          emaSat = relightEma * mergedStats.saturation + (1 - relightEma) * emaSat;
        }
        if (i === 0) {
          relightParams = { ...mergedStats };
        }
        const relightSmooth = { ...mergedStats, brightness: emaBright, saturation: emaSat };

        const cutoutPreRelight = cutout;
        cutout = await applyGlobalRelight(cutout, relightSmooth);

        if (usePyLap && maskMode !== 'fill') {
          const preP = path.join(processedDir, `_lap_pre_${String(i + 1).padStart(5, '0')}.png`);
          const postP = path.join(processedDir, `_lap_post_${String(i + 1).padStart(5, '0')}.png`);
          const mergedP = path.join(processedDir, `_lap_out_${String(i + 1).padStart(5, '0')}.png`);
          await fs.writeFile(preP, cutoutPreRelight);
          await fs.writeFile(postP, cutout);
          await laplacianMergePngPaths(preP, postP, mergedP);
          cutout = await fs.readFile(mergedP);
          await fs.unlink(preP).catch(() => {});
          await fs.unlink(postP).catch(() => {});
          await fs.unlink(mergedP).catch(() => {});
        } else if (hfPreserve) {
          cutout = await mergeRelightPreserveHighFreq(cutoutPreRelight, cutout, {
            sigma: hfSigma,
            detailStrength: hfDetail,
          });
        }

        let composed;
        if (maskMode === 'fill') {
          const rawFrame = await fs.readFile(framePath);
          let relitFull = await applyGlobalRelight(rawFrame, relightSmooth);
          if (usePyLap) {
            const preP = path.join(processedDir, `_lapf_pre_${String(i + 1).padStart(5, '0')}.png`);
            const postP = path.join(processedDir, `_lapf_post_${String(i + 1).padStart(5, '0')}.png`);
            const mergedP = path.join(processedDir, `_lapf_out_${String(i + 1).padStart(5, '0')}.png`);
            await fs.writeFile(preP, rawFrame);
            await fs.writeFile(postP, relitFull);
            await laplacianMergePngPaths(preP, postP, mergedP);
            relitFull = await fs.readFile(mergedP);
            await fs.unlink(preP).catch(() => {});
            await fs.unlink(postP).catch(() => {});
            await fs.unlink(mergedP).catch(() => {});
          } else if (hfPreserve) {
            relitFull = await mergeRelightPreserveHighFreq(rawFrame, relitFull, {
              sigma: hfSigma,
              detailStrength: hfDetail,
            });
          }
          composed = relitFull;
        } else {
          composed = await compositeCutoutOnBackground(bgLayer, cutout, outDims);
        }
        const outName = `out_${String(i + 1).padStart(5, '0')}.png`;
        await fs.writeFile(path.join(processedDir, outName), composed);
      }
      console.log('[switchx] composição concluída');

      const outDirFs = path.join(ROOT, 'outputs');
      await fs.mkdir(outDirFs, { recursive: true });
      const outFile = `output_${jobId}.mp4`;
      const outputPath = path.join(outDirFs, outFile);

      console.log('[switchx] encoding MP4 (ffmpeg)…');
      await createVideoFromFrames(processedDir, outputPath, extractFps);
      if (hasAudio) {
        try {
          await muxAudioFromSource(outputPath, videoPath, outputPath, {
            startSec: clipStart,
            durationSec: clipDur,
          });
          console.log('[switchx] áudio do source remuxado no MP4 (alinhado ao clip)');
        } catch (muxErr) {
          console.warn('[switchx] remux áudio ignorado:', muxErr?.message || muxErr);
        }
      }
      console.log('[switchx] mp4 encoded');
      let maskPreviewPublicUrl = '';
      if (previewMaskUrl) {
        const previewOutFile = outFile.replace('.mp4', '_mask_preview.png');
        const previewOutPath = path.join(outDirFs, previewOutFile);
        await fs.copyFile(previewMaskUrl, previewOutPath).catch(() => {});
        maskPreviewPublicUrl = `/outputs/${previewOutFile}`;
      }

      let sceneManifestUrl = '';
      const sceneOutDir = path.join(outDirFs, `scene_${jobId}`);
      try {
        await fs.mkdir(sceneOutDir, { recursive: true });
        await fs.copyFile(path.join(jobDir, 'scene', 'manifest.json'), path.join(sceneOutDir, 'manifest.json'));
        await fs.copyFile(
          path.join(jobDir, 'scene', 'temporal_cache.json'),
          path.join(sceneOutDir, 'temporal_cache.json'),
        );
        sceneManifestUrl = `/outputs/scene_${jobId}/manifest.json`;
      } catch (e) {
        console.warn('[switchx] cópia scene→outputs ignorada:', e?.message || e);
      }

      await safeRm(jobDir);
      await cleanupUploads();

      res.json({
        ok: true,
        videoUrl: `/outputs/${outFile}`,
        maskPreviewUrl: maskPreviewPublicUrl,
        sceneManifestUrl,
        sceneTemporalCacheUrl: sceneManifestUrl
          ? `/outputs/scene_${jobId}/temporal_cache.json`
          : '',
        hfPreserve,
        frames: frames.length,
        fps: extractFps,
        durationApprox: frames.length / extractFps,
        clipStartSec: clipStart,
        clipEndSec: clipEnd,
        nativeFps,
        approxNativeFrames,
        maskMode,
        model,
        quality,
        outputResolution,
        outWidth: outDims.width,
        outHeight: outDims.height,
      });
      console.log(`[switchx] done job=${jobId} output=${outFile}`);
    } catch (err) {
      console.error('[switchx]', err);
      await safeRm(jobDir);
      try {
        if (req.files?.video?.[0]?.path) await fs.unlink(req.files.video[0].path).catch(() => {});
        if (req.files?.background?.[0]?.path) await fs.unlink(req.files.background[0].path).catch(() => {});
        if (req.files?.reference?.[0]?.path) await fs.unlink(req.files.reference[0].path).catch(() => {});
        if (req.files?.maskUpload?.[0]?.path) await fs.unlink(req.files.maskUpload[0].path).catch(() => {});
      } catch {
        /* */
      }
      const msg = err.message || 'Erro no processamento.';
      const code =
        /Worker Python|Segmentação indisponível|pipeline worker|mask worker|Nenhum frame PNG/i.test(msg)
          ? 503
          : 500;
      res.status(code).json({ error: msg });
    }
  });

  return router;
}
