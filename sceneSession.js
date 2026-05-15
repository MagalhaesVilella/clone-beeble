/**
 * Persistência mínima de "sessão de cena" por job: índice temporal + placeholders
 * para fluxo / latent (camadas seguintes). Sem motor de render completo.
 */
import fs from 'fs/promises';
import path from 'path';

export function sceneDirForJob(jobDir) {
  return path.join(jobDir, 'scene');
}

/**
 * Garante pastas da sessão persistente.
 * @param {string} jobDir
 */
export async function ensureSceneDirs(jobDir) {
  const sd = sceneDirForJob(jobDir);
  await fs.mkdir(sd, { recursive: true });
  await fs.mkdir(path.join(sd, 'frames'), { recursive: true }).catch(() => {});
  return sd;
}

/**
 * Escreve manifest + cache temporal (placeholders). `frames` são paths absolutos.
 * @param {object} p
 * @param {string} p.jobDir
 * @param {string} p.jobId
 * @param {string[]} p.framePaths
 * @param {number} p.clipStartSec
 * @param {number} p.clipEndSec
 * @param {number} p.extractFps
 * @param {number} p.nativeFps
 * @param {{ width: number; height: number }} p.outDims
 * @param {{ duration?: number }} [p.videoProbe]
 */
export async function writeSceneManifest(p) {
  const sd = sceneDirForJob(p.jobDir);
  await ensureSceneDirs(p.jobDir);

  const rel = (abs) => path.relative(p.jobDir, abs).split(path.sep).join('/');

  const frames = (p.framePaths || []).map((fp, idx) => ({
    index: idx,
    path: rel(fp),
    time_sec: Number(p.clipStartSec) + idx / Math.max(1, Number(p.extractFps) || 1),
    decomposition: {
      mask: null,
      depth: null,
      normal: null,
      status: 'pending',
    },
    temporal: {
      flow_fwd: null,
      flow_bwd: null,
      flow_confidence: null,
      latent_path: null,
    },
  }));

  const manifest = {
    version: 1,
    philosophy: 'PRESERVE > ESTIMATE > RELIGHT > COMPOSITE',
    job_id: p.jobId,
    created_at: new Date().toISOString(),
    clip: {
      start_sec: p.clipStartSec,
      end_sec: p.clipEndSec,
      extract_fps: p.extractFps,
      native_fps: p.nativeFps,
    },
    canvas: { width: p.outDims.width, height: p.outDims.height },
    video: p.videoProbe || {},
    frames,
  };

  const temporalCache = {
    version: 1,
    note: 'Placeholders para RAFT / latent warp — preenchido em fases seguintes.',
    window: { max_frames: 8, ring_ptr: 0 },
    entries: frames.map((f) => ({
      index: f.index,
      flow_fwd: null,
      flow_bwd: null,
      latent: null,
    })),
  };

  await fs.writeFile(path.join(sd, 'manifest.json'), JSON.stringify(manifest, null, 2), 'utf-8');
  await fs.writeFile(path.join(sd, 'temporal_cache.json'), JSON.stringify(temporalCache, null, 2), 'utf-8');
}
