import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import multer from 'multer';
import { createSwitchxRouter } from './routes/switchx.js';
import { refreshPythonWorkerHealth, pythonWorkerState } from './services/pythonHealth.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname;
const PORT = process.env.PORT || 5050;
const require = createRequire(import.meta.url);
const { version: apiVersion } = require('./package.json');

await fs.mkdir(path.join(ROOT, 'tmp'), { recursive: true });
await fs.mkdir(path.join(ROOT, 'outputs'), { recursive: true });

/** Não bloquear o bind HTTP: o health-check Python (import torch) pode demorar >10s no Windows. */
void refreshPythonWorkerHealth().then((s) => {
  console.log('[switchx:mask] health-check inicial:', s?.message || s);
});

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, path.join(ROOT, 'tmp'));
  },
  filename: (_req, file, cb) => {
    const safe = `${Date.now()}_${file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_')}`;
    cb(null, safe);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 80 * 1024 * 1024 },
});

const app = express();
app.use(cors({ origin: true }));
app.use(express.json());

app.use('/outputs', express.static(path.join(ROOT, 'outputs')));

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'beeble-clone',
    version: apiVersion,
    python: {
      ok: pythonWorkerState.ok,
      message: pythonWorkerState.message,
      checkedAt: pythonWorkerState.checkedAt,
      bin: pythonWorkerState.pythonBin,
      imports: pythonWorkerState.imports,
    },
  });
});

app.use('/api/switchx', createSwitchxRouter({ upload }));

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Beeble-clone API em http://127.0.0.1:${PORT} (e http://localhost:${PORT})`);
});
