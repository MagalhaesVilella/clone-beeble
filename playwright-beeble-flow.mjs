/**
 * E2E: localhost:5178 — upload vídeo + imagem de fundo, Generate, consola + screenshot.
 */
import { chromium } from "playwright";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, "..");
const outDir = path.join(root, "tmp-e2e-playwright");
const base = process.env.BEEBLE_URL || "http://localhost:5178";

fs.mkdirSync(outDir, { recursive: true });

const videoPath = path.join(root, "test_video.mp4");
const imagePath = path.join(root, "test_bg.jpg");

async function download(url, dest) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
  fs.writeFileSync(dest, Buffer.from(await r.arrayBuffer()));
}

if (!fs.existsSync(imagePath) || fs.statSync(imagePath).size < 100) {
  await download("https://picsum.photos/seed/beeblebg/640/360.jpg", imagePath);
}

const consoleMsgs = [];
const pageErrors = [];

if (!fs.existsSync(videoPath) || fs.statSync(videoPath).size < 1000) {
  throw new Error(`Ficheiro não encontrado: ${videoPath}`);
}
if (!fs.existsSync(imagePath) || fs.statSync(imagePath).size < 100) {
  throw new Error(`Ficheiro não encontrado: ${imagePath}`);
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

page.on("console", (msg) => {
  consoleMsgs.push({ type: msg.type(), text: msg.text() });
});
page.on("pageerror", (err) => {
  pageErrors.push(String(err));
});

await page.goto(base, { waitUntil: "networkidle", timeout: 120000 });
await page.screenshot({ path: path.join(outDir, "01-initial.png"), fullPage: true });

// Source (secção 1): primeiro input de vídeo
await page.locator(".sw-sec").nth(0).locator('input[type="file"]').setInputFiles(videoPath);
await page.waitForTimeout(1500);
await page.getByRole("button", { name: /Confirmar trecho/i }).click({ timeout: 30000 });
await page.waitForTimeout(400);
await page.screenshot({ path: path.join(outDir, "02-after-video.png"), fullPage: true });

// Fundo: input dentro de .sw-btn-up
await page.locator(".sw-btn-up input[type=file]").setInputFiles(imagePath);
await page.waitForTimeout(800);
await page.screenshot({ path: path.join(outDir, "03-after-bg.png"), fullPage: true });

// Garantir tab AUTO selecionada na secção MASK
const autoTab = page.locator(".sw-sec").nth(1).getByRole("button", { name: /^Auto$/i });
if (await autoTab.count()) {
  await autoTab.click();
  await page.waitForTimeout(300);
}

const genBtn = page.getByRole("button", { name: /^Generate$/i });
await genBtn.waitFor({ state: "visible", timeout: 30000 });
const disabled = await genBtn.isDisabled();
if (disabled) {
  await page.screenshot({ path: path.join(outDir, "04-generate-disabled.png"), fullPage: true });
  const apiNote = await page.locator(".sw-err").textContent().catch(() => null);
  throw new Error(`Generate disabled. API banner: ${apiNote || "n/a"}`);
}

await genBtn.click();

let waitOutcome = "completed";
try {
  await page.waitForFunction(
    () => {
      const t = document.body?.innerText || "";
      return /Concluído|Erro:|Processing 100%|API offline/i.test(t);
    },
    null,
    { timeout: 180000 }
  );
} catch (err) {
  waitOutcome = `timeout-or-failure: ${String(err)}`;
}

await page.waitForTimeout(2000);
await page.screenshot({ path: path.join(outDir, "05-after-generate.png"), fullPage: true });

const statusText = await page.locator(".sw-st").textContent().catch(() => "");
const errBanner = await page.locator(".sw-err").textContent().catch(() => "");

const report = {
  base,
  statusLine: statusText?.trim() || null,
  errorBanner: errBanner?.trim() || null,
  waitOutcome,
  pageErrors,
  consoleMessages: consoleMsgs.filter(
    (m) => m.type === "error" || m.type === "warning" || /error|warn|fail/i.test(m.text)
  ),
  allConsoleTail: consoleMsgs.slice(-40),
  screenshots: [
    path.join(outDir, "01-initial.png"),
    path.join(outDir, "05-after-generate.png"),
  ],
};

fs.writeFileSync(path.join(outDir, "report.json"), JSON.stringify(report, null, 2), "utf8");
console.log(JSON.stringify(report, null, 2));

await browser.close();
