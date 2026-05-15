import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const backendDir = path.join(rootDir, 'backend');
const frontendDir = path.join(rootDir, 'frontend');
const backendEnvPath = path.join(backendDir, '.env');

function run(command, args, cwd = rootDir) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      stdio: 'inherit',
      shell: true,
    });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} ${args.join(' ')} falhou com código ${code}`));
    });
  });
}

async function freePortWindows(port) {
  const ps = [
    '$ErrorActionPreference = "SilentlyContinue"',
    `$pids = (Get-NetTCPConnection -LocalPort ${port} | Select-Object -ExpandProperty OwningProcess -Unique)`,
    'if ($pids) {',
    '  foreach ($pid in $pids) {',
    '    try { Stop-Process -Id $pid -Force } catch {}',
    '  }',
    '}',
  ].join('; ');
  await new Promise((resolve, reject) => {
    const child = spawn('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', ps], {
      cwd: rootDir,
      stdio: 'inherit',
      shell: false,
    });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Falha ao liberar porta ${port} (código ${code})`));
    });
  });
}

async function freeBusyPorts() {
  if (process.platform === 'win32') {
    console.log('[fix-all] Liberando portas 5050 e 5178...');
    await freePortWindows(5050).catch(() => {
      console.warn('[fix-all] Aviso: não foi possível liberar a porta 5050 automaticamente.');
    });
    await freePortWindows(5178).catch(() => {
      console.warn('[fix-all] Aviso: não foi possível liberar a porta 5178 automaticamente.');
    });
  }
}

async function ensureBackendEnv() {
  try {
    await fs.access(backendEnvPath);
  } catch {
    const envContent = ['PORT=5050', 'PYTHON_BIN=python', ''].join('\n');
    await fs.writeFile(backendEnvPath, envContent, 'utf-8');
  }
}

async function main() {
  console.log('\n[fix-all] Preparando ambiente (Node + Python)...\n');
  await ensureBackendEnv();
  await freeBusyPorts();

  await run('npm', ['install'], rootDir);
  await run('npm', ['install'], backendDir);
  await run('npm', ['install'], frontendDir);

  await run('python', ['-m', 'pip', 'install', '-r', 'requirements.txt'], backendDir);
  await run('python', ['-m', 'pip', 'install', 'einops', 'kornia', 'timm', 'rembg'], backendDir);

  console.log('\n[fix-all] Ambiente pronto. Execute: npm run dev\n');
}

main().catch((err) => {
  console.error('\n[fix-all] Erro:', err.message);
  process.exit(1);
});
