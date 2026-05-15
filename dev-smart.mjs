import net from 'node:net';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');

function isPortFree(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(false));
    server.once('listening', () => {
      server.close(() => resolve(true));
    });
    server.listen(port);
  });
}

async function findFreePort(start) {
  for (let p = start; p < start + 50; p += 1) {
    // eslint-disable-next-line no-await-in-loop
    if (await isPortFree(p)) return p;
  }
  throw new Error(`Não foi possível encontrar porta livre a partir de ${start}`);
}

async function main() {
  const apiPort = await findFreePort(5050);
  const webPort = await findFreePort(5178);

  console.log(`[dev-smart] API porta: ${apiPort}`);
  console.log(`[dev-smart] WEB porta: ${webPort}`);

  const env = {
    ...process.env,
    PORT: String(apiPort),
    VITE_PORT: String(webPort),
    VITE_API_PORT: String(apiPort),
  };

  const api = spawn('npm', ['run', 'dev', '--prefix', 'backend'], {
    cwd: rootDir,
    shell: true,
    stdio: 'inherit',
    env,
  });

  const web = spawn('npm', ['run', 'dev', '--prefix', 'frontend'], {
    cwd: rootDir,
    shell: true,
    stdio: 'inherit',
    env,
  });

  const stopAll = () => {
    try {
      api.kill('SIGTERM');
    } catch {
      /* ignore */
    }
    try {
      web.kill('SIGTERM');
    } catch {
      /* ignore */
    }
  };

  process.on('SIGINT', stopAll);
  process.on('SIGTERM', stopAll);

  let exited = false;
  const onExit = (code) => {
    if (exited) return;
    exited = true;
    stopAll();
    process.exit(code || 0);
  };
  api.on('close', onExit);
  web.on('close', onExit);
}

main().catch((err) => {
  console.error('[dev-smart] Erro:', err.message);
  process.exit(1);
});
