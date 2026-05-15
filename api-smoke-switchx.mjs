import fs from "fs";
import path from "path";

const root = process.cwd();
const videoPath = path.join(root, "test_video.mp4");
const bgPath = path.join(root, "test_bg.jpg");

if (!fs.existsSync(videoPath) || !fs.existsSync(bgPath)) {
  throw new Error("Arquivos de teste não encontrados (test_video.mp4/test_bg.jpg).");
}

const form = new FormData();
form.append("video", new Blob([fs.readFileSync(videoPath)]), "test_video.mp4");
form.append("background", new Blob([fs.readFileSync(bgPath)]), "test_bg.jpg");
form.append("reference", new Blob([fs.readFileSync(bgPath)]), "test_bg.jpg");
form.append("maskMode", "auto");
form.append("model", "google");
form.append("clipStartSec", "0");
form.append("clipEndSec", "4");

const started = Date.now();
const res = await fetch("http://127.0.0.1:5050/api/switchx", {
  method: "POST",
  body: form,
});
const txt = await res.text();
let json = null;
try {
  json = JSON.parse(txt);
} catch {
  json = { raw: txt };
}

console.log(
  JSON.stringify(
    {
      okHttp: res.ok,
      status: res.status,
      elapsedMs: Date.now() - started,
      response: json,
    },
    null,
    2
  )
);
