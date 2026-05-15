from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional

# Timeout por defeito para downloads HF / carregamento BiRefNet (evita bloqueio indefinido).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from transformers import AutoModelForImageSegmentation

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODNET_ONNX = MODELS_DIR / "modnet_photographic.onnx"
MODNET_URL = "https://github.com/yakhyo/modnet/releases/download/weights/modnet_photographic.onnx"

FRAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


def _mlog(msg: str) -> None:
    print(f"[switchx:mask] {msg}", file=sys.stderr, flush=True)


def _ensure_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _download_modnet_if_missing():
    _ensure_models_dir()
    if MODNET_ONNX.exists():
        _mlog(f"MODNet ONNX encontrado: {MODNET_ONNX}")
        return
    import urllib.request

    _mlog(f"MODNet em falta — a descarregar para {MODNET_ONNX} …")
    try:
        urllib.request.urlretrieve(MODNET_URL, MODNET_ONNX.as_posix())
        _mlog("MODNet descarregado com sucesso.")
    except Exception as exc:  # noqa: BLE001
        _mlog(f"MODNet download falhou: {exc}. Baixa manualmente de {MODNET_URL} para {MODNET_ONNX}")
        raise


def _list_frame_pngs(in_dir: Path) -> list:
    frames = [p for p in in_dir.iterdir() if p.is_file() and FRAME_RE.match(p.name)]
    return sorted(frames, key=lambda p: int(FRAME_RE.match(p.name).group(1)))


class BiRefNetSegmenter:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.error = None
        timeout = float(os.environ.get("BIREFNET_LOAD_TIMEOUT_SEC", "180"))
        holder: dict = {}

        def _load_worker():
            try:
                _mlog("BiRefNet: a carregar ZhengPeng7/BiRefNet …")
                m = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet", trust_remote_code=True
                )
                m.to(device)
                m.eval()
                holder["model"] = m
            except Exception as exc:  # noqa: BLE001
                holder["err"] = str(exc)

        th = threading.Thread(target=_load_worker, daemon=True, name="birefnet-load")
        th.start()
        th.join(timeout=timeout)
        if th.is_alive():
            self.error = (
                f"BiRefNet: timeout após {timeout}s (download/carregamento Hugging Face). "
                "O pipeline usa MODNet/rembg. Para pré-cache: HF_HOME ou "
                "`huggingface-cli download ZhengPeng7/BiRefNet`. Opcional: BIREFNET_LOAD_TIMEOUT_SEC."
            )
            _mlog(self.error)
            self.model = None
            return
        if holder.get("err"):
            self.error = holder["err"]
            self.model = None
            _mlog(f"BiRefNet falhou ({self.error}); a tentar MODNet/rembg.")
        else:
            self.model = holder.get("model")
            if self.model is not None:
                _mlog("BiRefNet: carregado.")

    def get_mask(self, image_pil):
        if self.model is None:
            return None
        image = image_pil.convert("RGB").resize((1024, 1024), Image.BICUBIC)
        x = np.asarray(image).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            pred = out[-1] if isinstance(out, (list, tuple)) else out
            pred = torch.sigmoid(pred)[0, 0].detach().cpu().numpy()
        return np.clip(pred, 0.0, 1.0)


class MODNetRefiner:
    def __init__(self):
        self.enabled = False
        self.session = None
        self.input_name = None
        self.error = None
        if ort is None:
            self.error = "onnxruntime não disponível"
            _mlog("MODNet: onnxruntime não importado.")
            return
        try:
            _download_modnet_if_missing()
            self.session = ort.InferenceSession(MODNET_ONNX.as_posix(), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.enabled = True
            _mlog("MODNet: sessão ONNX inicializada (CPU).")
        except Exception as exc:
            self.error = str(exc)
            self.enabled = False
            _mlog(f"MODNet falhou: {exc}")

    def refine(self, image_pil):
        if not self.enabled:
            return None
        rgb = np.asarray(image_pil.convert("RGB")).astype(np.float32)
        h, w = rgb.shape[:2]
        ref_size = 512
        if w >= h:
            rh = ref_size
            rw = int(w / max(h, 1) * ref_size)
        else:
            rw = ref_size
            rh = int(h / max(w, 1) * ref_size)
        rw = max(32, rw - rw % 32)
        rh = max(32, rh - rh % 32)
        resized = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((rw, rh), Image.BICUBIC)).astype(np.float32)
        x = resized / 255.0
        x = (x - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
        out = self.session.run(None, {self.input_name: x})
        matte = out[-1][0, 0]
        matte_img = Image.fromarray(np.clip(matte * 255.0, 0, 255).astype(np.uint8), mode="L").resize((w, h), Image.BICUBIC)
        return np.asarray(matte_img).astype(np.float32) / 255.0


class RembgFallback:
    def __init__(self):
        self.enabled = rembg_remove is not None
        self.error = None if self.enabled else "rembg não instalado"
        if self.enabled:
            _mlog("rembg (u2net): disponível como fallback.")
        else:
            _mlog("rembg: não instalado.")

    def get_mask(self, image_pil):
        if not self.enabled:
            return None
        try:
            buf = BytesIO()
            image_pil.convert("RGB").save(buf, format="PNG")
            mask_bytes = rembg_remove(buf.getvalue(), only_mask=True)
            mask = Image.open(BytesIO(mask_bytes)).convert("L").resize(image_pil.size, Image.BICUBIC)
            return np.asarray(mask).astype(np.float32) / 255.0
        except Exception as exc:
            self.error = str(exc)
            return None


def _selection_mask(size, points):
    """Pontos em pixels (0..w-1, 0..h-1) ou normalizados em [0,1] se x,y <= 1."""
    w, h = size
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    radius = max(20, int(min(w, h) * 0.06))
    for p in points:
        if not isinstance(p, dict):
            continue
        xf = float(p.get("x", 0))
        yf = float(p.get("y", 0))
        if xf <= 1.0 and yf <= 1.0 and xf >= 0 and yf >= 0:
            x = int(xf * (w - 1)) if w > 1 else 0
            y = int(yf * (h - 1)) if h > 1 else 0
        else:
            x = int(np.clip(xf, 0, w - 1))
            y = int(np.clip(yf, 0, h - 1))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=max(6, radius // 3)))
    return np.asarray(canvas).astype(np.float32) / 255.0


def _remove_small_islands(alpha, min_ratio=0.002):
    """Remove blobs isolados menores que min_ratio da área total."""
    try:
        from scipy import ndimage

        binary = (alpha > 0.4).astype(np.uint8)
        labeled, num = ndimage.label(binary)
        if num == 0:
            return alpha
        total_pixels = alpha.shape[0] * alpha.shape[1]
        min_size = int(total_pixels * min_ratio)
        for label_id in range(1, num + 1):
            if np.sum(labeled == label_id) < min_size:
                alpha[labeled == label_id] = 0.0
        return alpha
    except ImportError:
        return alpha


def _smooth_alpha(alpha):
    alpha_img = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return np.asarray(alpha_img).astype(np.float32) / 255.0


def _morph_repair_alpha(alpha_f: np.ndarray) -> np.ndarray:
    """Fecho morfológico + buracos + leve abertura: fecha falhas no torso/cabelo (scipy opcional)."""
    try:
        from scipy import ndimage

        b = (alpha_f > 0.26).astype(np.uint8)
        b = ndimage.binary_closing(b, iterations=3)
        b = ndimage.binary_fill_holes(b)
        b = ndimage.binary_opening(b, iterations=1)
        closed = b.astype(np.float32)
        return np.clip(np.maximum(alpha_f, closed * 0.62), 0.0, 1.0)
    except Exception:
        return alpha_f


def _floor_weak_alpha(alpha_f: np.ndarray, stage_hi: Optional[np.ndarray], floor: float = 0.08) -> np.ndarray:
    """Se o modelo principal deixou alpha globalmente fraco, sobe suavemente onde há sinal em stage_hi."""
    if stage_hi is None or alpha_f.size == 0:
        return alpha_f
    if float(np.mean(alpha_f)) >= 0.04:
        return alpha_f
    boost = np.clip(stage_hi.astype(np.float32) * 0.75, 0.0, 1.0)
    mask = boost > 0.18
    out = np.where(mask, np.maximum(alpha_f, np.maximum(boost, floor)), alpha_f)
    _mlog(f"alpha global fraco (média={float(np.mean(alpha_f)):.4f}) — boost conservador aplicado.")
    return np.clip(out, 0.0, 1.0)


def _binary_preview(alpha):
    """Preview estritamente binário: branco sujeito / preto fundo."""
    return ((alpha >= 0.5).astype(np.uint8) * 255)


def _export_mask_u8(alpha: np.ndarray) -> np.ndarray:
    """
    Converte alpha float [0,1] para uint8 (L).
    Se SWITCHX_MASK_EXPORT_THRESHOLD estiver definido (0–255), aplica binário:
    pixel >= T → 255, senão 0 (equivalente ao cv2.THRESH_BINARY).
    Sem variável: mantém tons de cinza (penas); o Sharp usa isto como canal alpha.
    """
    u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    raw = os.environ.get("SWITCHX_MASK_EXPORT_THRESHOLD", "").strip()
    if not raw:
        return u8
    try:
        t = int(raw, 10)
    except ValueError:
        return u8
    if 0 <= t <= 255:
        return ((u8 >= t).astype(np.uint8)) * 255
    return u8


def _compute_alpha(image, mode, points_json, user_mask_path, birefnet, modnet, rembg):
    w, h = image.size

    if mode == "fill":
        alpha = np.ones((h, w), dtype=np.float32)
    elif mode == "upload" and user_mask_path:
        user_mask = Image.open(user_mask_path).convert("L").resize((w, h), Image.BICUBIC)
        alpha = np.asarray(user_mask).astype(np.float32) / 255.0
    else:
        stage1 = birefnet.get_mask(image)
        if stage1 is not None:
            stage1 = np.asarray(
                Image.fromarray((stage1 * 255).astype(np.uint8), mode="L").resize((w, h), Image.BICUBIC)
            ).astype(np.float32) / 255.0
        stage2 = modnet.refine(image)
        if stage1 is None and stage2 is None:
            stage3 = rembg.get_mask(image)
            if stage3 is None:
                stage1 = np.ones((h, w), dtype=np.float32)
                stage2 = stage1
            else:
                stage1 = stage3
                stage2 = stage3
        elif stage1 is None:
            stage1 = stage2
        elif stage2 is None:
            stage2 = stage1

        if stage1 is not None and stage2 is not None:
            smax = np.maximum(stage1, stage2)
            sblend = np.clip(0.5 * stage1 + 0.5 * stage2, 0.0, 1.0)
            core_mask = (smax > 0.4).astype(np.float32)
            transition = ((smax >= 0.12) & (smax <= 0.88)).astype(np.float32)
            alpha = np.clip(
                core_mask * smax * (1.0 - 0.35 * transition)
                + (1.0 - core_mask * (1.0 - transition)) * np.maximum(sblend, smax * 0.65),
                0.0,
                1.0,
            )
        else:
            alpha = stage1 if stage1 is not None else np.ones((h, w), dtype=np.float32)

        if mode == "select" and points_json:
            points = json.loads(points_json) if isinstance(points_json, str) else points_json
            pick = _selection_mask((w, h), points if isinstance(points, list) else [])
            alpha = np.clip(alpha * (0.4 + 0.6 * pick), 0.0, 1.0)

        alpha = _floor_weak_alpha(alpha, stage1 if stage1 is not None else stage2)

    alpha = _remove_small_islands(alpha)
    alpha = _smooth_alpha(alpha)
    alpha = _morph_repair_alpha(alpha)
    if float(np.mean(alpha)) < 0.02 and mode not in ("fill", "upload"):
        _mlog("alpha ainda quase nulo após reparo — tentativa rembg única.")
        stage_r = rembg.get_mask(image)
        if stage_r is not None:
            alpha = np.clip(np.maximum(alpha, stage_r.astype(np.float32)), 0.0, 1.0)

    if mode not in ("fill",):
        try:
            from mask_refine_cv import refine_mask_alpha

            rgb_a = np.asarray(image.convert("RGB"))
            alpha = refine_mask_alpha(rgb_a, alpha)
        except Exception as exc:  # noqa: BLE001
            _mlog(f"refine_mask_alpha: {exc}")

    return alpha


def run(input_path, output_path, mode, points_json=None, user_mask_path=None, preview_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(input_path).convert("RGB")
    needs_model = not (mode == "fill" or (mode == "upload" and user_mask_path))
    birefnet = BiRefNetSegmenter(device) if needs_model else BiRefNetSegmenter.__new__(BiRefNetSegmenter)
    modnet = MODNetRefiner() if needs_model else MODNetRefiner.__new__(MODNetRefiner)
    rembg = RembgFallback()
    if not needs_model:
        birefnet.model = None
        birefnet.error = ""
        modnet.enabled = False
        modnet.error = ""
    if needs_model and birefnet.model is None and not modnet.enabled and not rembg.enabled:
        raise RuntimeError(
            "Segmentação indisponível: nenhum modelo carregou (BiRefNet, MODNet, rembg). "
            "Verifica os logs [switchx:mask] do backend."
        )
    alpha = _compute_alpha(image, mode, points_json, user_mask_path, birefnet, modnet, rembg)
    out = _export_mask_u8(alpha)
    Image.fromarray(out, mode="L").save(str(output_path), format="PNG")

    if preview_path:
        Image.fromarray(_binary_preview(alpha), mode="L").save(str(preview_path), format="PNG")
    return {
        "birefnet_ok": (birefnet.model is not None) if needs_model else False,
        "modnet_ok": modnet.enabled if needs_model else False,
        "rembg_ok": rembg.enabled,
        "birefnet_error": birefnet.error or "",
        "modnet_error": modnet.error or "",
        "rembg_error": rembg.error or "",
    }

def run_batch(input_dir, output_dir, mode, points_json=None, user_mask_path=None, preview_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    needs_model = not (mode == "fill" or (mode == "upload" and user_mask_path))
    birefnet = BiRefNetSegmenter(device) if needs_model else BiRefNetSegmenter.__new__(BiRefNetSegmenter)
    modnet = MODNetRefiner() if needs_model else MODNetRefiner.__new__(MODNetRefiner)
    rembg = RembgFallback()
    if not needs_model:
        birefnet.model = None
        birefnet.error = ""
        modnet.enabled = False
        modnet.error = ""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "_batch_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if needs_model and birefnet.model is None and not modnet.enabled and not rembg.enabled:
        raise RuntimeError(
            "Segmentação indisponível: nenhum modelo carregou (BiRefNet, MODNet, rembg). "
            "Verifica os logs [switchx:mask] do backend e as dependências Python."
        )
    frames = _list_frame_pngs(in_dir)
    if not frames:
        raise RuntimeError(
            f"Nenhum frame PNG encontrado em {in_dir} (esperado frame_00001.png, …). "
            "Confirma a extração FFmpeg e o padrão de nomes."
        )
    for i, frame in enumerate(frames, start=1):
        image = Image.open(frame).convert("RGB")
        alpha = _compute_alpha(image, mode, points_json, user_mask_path, birefnet, modnet, rembg)
        out = _export_mask_u8(alpha)
        out_file = mask_dir / f"mask_{i:05d}.png"
        Image.fromarray(out, mode="L").save(str(out_file), format="PNG")
        if i == 1:
            m0 = float(np.mean(out)) / 255.0
            _mlog(f"run_batch: 1.ª máscara mean={m0:.4f} (0–1); ficheiro {out_file.name}")
            if m0 < 0.05:
                _mlog("run_batch: AVISO — 1.ª máscara quase preta; o batch pode estar errado (paths/cores/modelo).")
        if i == 1 and preview_path:
            Image.fromarray(_binary_preview(alpha), mode="L").save(str(preview_path), format="PNG")
    return {
        "birefnet_ok": (birefnet.model is not None) if needs_model else False,
        "modnet_ok": modnet.enabled if needs_model else False,
        "rembg_ok": rembg.enabled,
        "birefnet_error": birefnet.error or "",
        "modnet_error": modnet.error or "",
        "rembg_error": rembg.error or "",
        "frames": len(frames),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--points", default="")
    parser.add_argument("--user-mask", default="")
    parser.add_argument("--preview", default="")
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    if args.input_dir and args.output_dir:
        info = run_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            points_json=args.points or None,
            user_mask_path=args.user_mask or None,
            preview_path=args.preview or None,
        )
        if args.report:
            Path(args.report).write_text(json.dumps(info), encoding="utf-8")
    else:
        if not args.input or not args.output:
            raise SystemExit("For single mode, --input and --output are required.")
        info = run(
            input_path=args.input,
            output_path=args.output,
            mode=args.mode,
            points_json=args.points or None,
            user_mask_path=args.user_mask or None,
            preview_path=args.preview or None,
        )
        if args.report:
            Path(args.report).write_text(json.dumps(info), encoding="utf-8")
