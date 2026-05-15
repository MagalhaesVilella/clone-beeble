"""
Worker opcional SAM 2.1 (vídeo): propagação temporal, contrato alinhado com pipeline.py.

Requisitos: repo oficial `facebookresearch/sam2` instalado (`pip install -e ".[demo]"`)
e variável SAM2_CHECKPOINT com caminho absoluto para o .pt (ex.: sam2.1_hiera_small.pt).

O carregador de vídeo do SAM2 espera JPEGs com nome só numérico (ex.: 00001.jpg), não PNG frame_*.png.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

FRAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)

# Defaults alinhados ao pacote SAM 2.1 (Hydra resolve a partir do pacote `sam2` instalado).
_DEFAULT_CFG = os.environ.get("SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml")
_DEFAULT_CKPT = os.environ.get("SAM2_CHECKPOINT", "")


def _mlog(msg: str) -> None:
    print(f"[switchx:mask] sam2: {msg}", file=sys.stderr, flush=True)


def _list_frame_pngs(in_dir: Path) -> list[Path]:
    frames = [p for p in in_dir.iterdir() if p.is_file() and FRAME_RE.match(p.name)]
    return sorted(frames, key=lambda p: int(FRAME_RE.match(p.name).group(1)))


def _prepare_jpeg_sequence(frames: list[Path], jpg_dir: Path) -> tuple[int, int]:
    """Converte PNGs para JPEG com nomes 00001.jpg … exigidos por load_video_frames."""
    jpg_dir.mkdir(parents=True, exist_ok=True)
    w0 = h0 = 0
    for i, png in enumerate(frames, start=1):
        img = Image.open(png).convert("RGB")
        w0, h0 = img.size
        out = jpg_dir / f"{i:05d}.jpg"
        img.save(out, format="JPEG", quality=95)
    return w0, h0


def _masks_to_u8(video_res_masks: torch.Tensor) -> np.ndarray:
    """Logits ou probabilidades [N?, H, W] ou [N,1,H,W] → uint8 L 0/255."""
    t = video_res_masks.detach().float().cpu()
    while t.dim() > 3 and t.shape[-3] == 1:
        t = t.squeeze(-3)
    if t.dim() == 4:
        t = t.squeeze(1)
    if t.dim() == 3:
        combined = (t > 0.0).any(dim=0)
    elif t.dim() == 2:
        combined = t > 0.0
    else:
        raise ValueError(f"forma de máscara inesperada: {tuple(t.shape)}")
    return (combined.numpy().astype(np.uint8) * 255)


def segment_video_sam2(
    frames_dir: str,
    masks_dir: str,
    mode: str = "auto",
    points: list | None = None,
    prompt_text: str = "person",
    preview_path: str | None = None,
    model_cfg: str | None = None,
    checkpoint: str | None = None,
) -> dict:
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        return {
            "sam2_ok": False,
            "error": f"pacote sam2 não disponível: {exc}",
        }

    cfg = model_cfg or _DEFAULT_CFG
    ckpt = (checkpoint or _DEFAULT_CKPT or "").strip()
    if not ckpt or not Path(ckpt).is_file():
        return {
            "sam2_ok": False,
            "error": "SAM2_CHECKPOINT em falta ou ficheiro inexistente (caminho absoluto para o .pt).",
            "model_config": cfg,
        }

    in_dir = Path(frames_dir)
    out_dir = Path(masks_dir)
    mask_dir = out_dir / "_batch_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    frames = _list_frame_pngs(in_dir)
    if not frames:
        return {
            "sam2_ok": False,
            "error": f"nenhum frame_*.png em {in_dir}",
        }

    jpg_root = Path(tempfile.mkdtemp(prefix="sam2-jpg-", dir=str(out_dir.parent)))
    try:
        w, h = _prepare_jpeg_sequence(frames, jpg_root)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _mlog(f"a carregar SAM2 ({cfg}) em {device} …")
        predictor = build_sam2_video_predictor(cfg, ckpt, device=device)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else contextlib.nullcontext()
        )
        with amp_ctx:
            state = predictor.init_state(video_path=str(jpg_root))

            obj_id = 1
            labels_pos = np.array([1], dtype=np.int32)

            if mode == "select" and points:
                for pi, pt in enumerate(points):
                    if not isinstance(pt, dict):
                        continue
                    px = float(pt.get("x", 0))
                    py = float(pt.get("y", 0))
                    pts = np.array([[px, py]], dtype=np.float32)
                    labs = labels_pos
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=obj_id,
                        points=pts,
                        labels=labs,
                        clear_old_points=(pi == 0),
                    )
            else:
                pts = np.array([[float(w // 2), float(h // 2)]], dtype=np.float32)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=pts,
                    labels=labels_pos,
                    clear_old_points=True,
                )

            frame_count = 0
            for frame_idx, _obj_ids, video_res_masks in predictor.propagate_in_video(state):
                combined = _masks_to_u8(video_res_masks)
                if combined.shape != (h, w):
                    combined = np.array(
                        Image.fromarray(combined, mode="L").resize((w, h), Image.BILINEAR)
                    )
                mask_path = mask_dir / f"mask_{frame_idx + 1:05d}.png"
                Image.fromarray(combined, mode="L").save(mask_path, format="PNG")
                frame_count += 1

        if preview_path and frame_count:
            first = mask_dir / "mask_00001.png"
            if first.is_file():
                shutil.copyfile(first, preview_path)

        model_name = Path(ckpt).stem
        return {
            "sam2_ok": True,
            "frames_processed": frame_count,
            "model": model_name,
            "model_config": cfg,
            "prompt_text": prompt_text,
            "prompt_note": "SAM2 base não tem prompt textual; usado apenas ponto(s) no frame 0.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"sam2_ok": False, "error": str(exc)}
    finally:
        shutil.rmtree(jpg_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--points", default="")
    parser.add_argument("--prompt-text", default="person")
    parser.add_argument("--preview", default="")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    pts = None
    if args.points:
        try:
            pts = json.loads(args.points)
        except json.JSONDecodeError:
            pts = None

    report = segment_video_sam2(
        frames_dir=args.frames_dir,
        masks_dir=args.masks_dir,
        mode=args.mode,
        points=pts if isinstance(pts, list) else None,
        prompt_text=args.prompt_text or "person",
        preview_path=args.preview or None,
    )

    Path(args.report).write_text(json.dumps(report), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
