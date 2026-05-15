"""
Worker opcional Robust Video Matting (RVM, PeterL1n).
Contrato alinhado com pipeline.py / sam2: --frames-dir, --masks-dir, --mode, --points, --report, --preview.

Carregamento: torch.hub (descarrega código + pesos na 1.ª execução) ou RVM_CHECKPOINT + RVM_VARIANT.
Entrada: frame_00001.png, … (mesmo padrão que extractFrames).
Saída: _batch_masks/mask_00001.png … + report.json com rvm_ok.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms

FRAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


def _mlog(msg: str) -> None:
    print(f"[switchx:mask] rvm: {msg}", file=sys.stderr, flush=True)


def _list_frame_pngs(in_dir: Path) -> list[Path]:
    frames = [p for p in in_dir.iterdir() if p.is_file() and FRAME_RE.match(p.name)]
    return sorted(frames, key=lambda p: int(FRAME_RE.match(p.name).group(1)))


def auto_downsample_ratio(h: int, w: int) -> float:
    return float(min(512 / max(h, w), 1.0))


def _selection_mask(size: tuple[int, int], points: list) -> np.ndarray:
    """Igual à ideia do mask_pipeline: pesos 0..1 por cliques (pixels ou 0..1)."""
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


def _export_u8(alpha_f: np.ndarray) -> np.ndarray:
    u8 = np.clip(alpha_f * 255.0, 0, 255).astype(np.uint8)
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


def _load_model(device: str, variant: str, checkpoint: str | None):
    variant = (variant or "mobilenetv3").lower()
    if variant not in ("mobilenetv3", "resnet50"):
        variant = "mobilenetv3"
    ckpt = (checkpoint or "").strip()
    _mlog(f"a carregar RVM ({variant}) via torch.hub …")
    if ckpt and Path(ckpt).is_file():
        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            variant,
            pretrained=False,
            trust_repo=True,
        )
        try:
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd)
    else:
        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            variant,
            pretrained=True,
            trust_repo=True,
        )
    return model.eval().to(device), variant


def segment_video_rvm(
    frames_dir: str,
    masks_dir: str,
    mode: str = "auto",
    points: list | None = None,
    preview_path: str | None = None,
    variant: str | None = None,
    checkpoint: str | None = None,
) -> dict:
    in_dir = Path(frames_dir)
    out_dir = Path(masks_dir)
    mask_dir = out_dir / "_batch_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    frames = _list_frame_pngs(in_dir)
    if not frames:
        return {"rvm_ok": False, "error": f"nenhum frame_*.png em {in_dir}"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = variant or os.environ.get("RVM_VARIANT", "mobilenetv3")
    ck = checkpoint or os.environ.get("RVM_CHECKPOINT", "")

    try:
        model, v_used = _load_model(device, v, ck)
    except Exception as exc:  # noqa: BLE001
        return {"rvm_ok": False, "error": f"falha a carregar RVM: {exc}"}

    to_tensor = transforms.ToTensor()
    pick = None
    if mode == "select" and points:
        w0, h0 = Image.open(frames[0]).convert("RGB").size
        pick = _selection_mask((w0, h0), points)

    dtype = next(model.parameters()).dtype
    n = 0
    with torch.no_grad():
        rec: list = [None] * 4
        for frame_path in frames:
            img = Image.open(frame_path).convert("RGB")
            w, h = img.size
            t = to_tensor(img).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
            dr = auto_downsample_ratio(h, w)
            fgr, pha, *rec = model(t, *rec, downsample_ratio=dr)
            alpha = pha[0, 0, 0].float().cpu().numpy()
            if pick is not None:
                if pick.shape != alpha.shape:
                    pick_r = np.array(
                        Image.fromarray((pick * 255).astype(np.uint8), mode="L").resize(
                            (alpha.shape[1], alpha.shape[0]), Image.BILINEAR
                        )
                    ).astype(np.float32) / 255.0
                else:
                    pick_r = pick
                alpha = np.clip(alpha * (0.4 + 0.6 * pick_r), 0.0, 1.0)
            u8 = _export_u8(alpha)
            n += 1
            out_png = mask_dir / f"mask_{n:05d}.png"
            Image.fromarray(u8, mode="L").save(out_png, format="PNG")

    if preview_path and n:
        first = mask_dir / "mask_00001.png"
        if first.is_file():
            shutil.copyfile(first, preview_path)

    return {
        "rvm_ok": True,
        "frames_processed": n,
        "model": f"rvm_{v_used}",
        "device": device,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--points", default="")
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--preview", default="")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    pts = None
    if args.points:
        try:
            pts = json.loads(args.points)
        except json.JSONDecodeError:
            pts = None

    try:
        report = segment_video_rvm(
            frames_dir=args.frames_dir,
            masks_dir=args.masks_dir,
            mode=args.mode,
            points=pts if isinstance(pts, list) else None,
            preview_path=args.preview or None,
        )
    except Exception as exc:  # noqa: BLE001
        report = {"rvm_ok": False, "error": str(exc)}

    Path(args.report).write_text(json.dumps(report), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
