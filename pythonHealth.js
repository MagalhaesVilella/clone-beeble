/**
 * Health-check do interpretador Python usado pelo worker de máscaras.
 */
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HEALTH_SCRIPT = path.join(__dirname, 'python', 'health_imports.py');

export const pythonWorkerState = {
  ok: false,
  message: '',
  checkedAt: null,
  pythonBin: process.env.PYTHON_BIN || (process.platform === 'win32' ? 'python' : 'python3'),
  imports: null,
};

export function getPythonBin() {
  return process.env.PYTHON_BIN || (process.platform === 'win32' ? 'python' : 'python3');
}

export function isPythonWorkerReady() {
  return pythonWorkerState.ok === true;
}

export async function refreshPythonWorkerHealth() {
  const bin = getPythonBin();
  pythonWorkerState.pythonBin = bin;
  pythonWorkerState.checkedAt = new Date().toISOString();
  return new Promise((resolve) => {
    const p = spawn(bin, [HEALTH_SCRIPT], { windowsHide: true });
    let out = '';
    let err = '';
    p.stdout?.on('data', (d) => {
      out += d.toString();
    });
    p.stderr?.on('data', (d) => {
      err += d.toString();
    });
    p.on('error', (e) => {
      pythonWorkerState.ok = false;
      pythonWorkerState.message = `Não foi possível executar ${bin}: ${e.message}`;
      pythonWorkerState.imports = null;
      console.error('[switchx:mask] health-check spawn falhou:', e.message);
      resolve(pythonWorkerState);
    });
    p.on('close', (code) => {
      if (code !== 0) {
        pythonWorkerState.ok = false;
        pythonWorkerState.message =
          err.trim() || out.trim() || `Health-check Python terminou com código ${code}. Instala: pip install -r backend/requirements.txt`;
        pythonWorkerState.imports = null;
        console.error('[switchx:mask] health-check falhou:', pythonWorkerState.message);
        resolve(pythonWorkerState);
        return;
      }
      try {
        const line = out.trim().split('\n').filter(Boolean).pop();
        const j = JSON.parse(line || '{}');
        pythonWorkerState.ok = !!j.ok;
        pythonWorkerState.imports = j;
        pythonWorkerState.message = j.ok
          ? `Python OK (torch ${j.torch || '?'})`
          : j.error || 'Import falhou';
        if (!j.ok) console.error('[switchx:mask] health-check:', pythonWorkerState.message);
        else console.log('[switchx:mask] health-check:', pythonWorkerState.message);
      } catch (e) {
        pythonWorkerState.ok = false;
        pythonWorkerState.message = `Resposta inválida do Python: ${out.slice(0, 200)}`;
        pythonWorkerState.imports = null;
        console.error('[switchx:mask] health-check parse:', e.message);
      }
      resolve(pythonWorkerState);
    });
  });
}
