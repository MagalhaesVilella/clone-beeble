"""
Pós-processamento de máscaras alpha: blur, guided filter (se disponível) e abertura morfológica.
Controlar com SWITCHX_MASK_REFINE_CV=0 para desactivar.
"""
from __future__ import annotations

import os

import numpy as np


def refine_mask_alpha(rgb: np.ndarray, alpha_f: np.ndarray) -> np.ndarray:
    """
    :param rgb: uint8 (H, W, 3) RGB
    :param alpha_f: float32 (H, W) em [0, 1]
    :return: float32 (H, W) em [0, 1]
    """
    v = os.environ.get("SWITCHX_MASK_REFINE_CV", "1").lower()
    if v in ("0", "false", "no"):
        return alpha_f

    try:
        import cv2
    except Exception:
        return alpha_f

    if rgb.shape[:2] != alpha_f.shape[:2]:
        return alpha_f

    a = np.clip(alpha_f.astype(np.float32), 0.0, 1.0)
    m = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    # 1) GaussianBlur 5x5
    m = cv2.GaussianBlur(m, (5, 5), 0)
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mf = m.astype(np.float32) / 255.0

    # 2) Guided filter (contrib) ou bilateral como fallback
    refined = mf
    try:
        import cv2.ximgproc as xi  # type: ignore[attr-defined]

        refined = xi.guidedFilter(
            guide=guide,
            src=mf,
            radius=int(os.environ.get("SWITCHX_GUIDED_RADIUS", "8")),
            eps=float(os.environ.get("SWITCHX_GUIDED_EPS", "1e-3")),
        )
    except Exception:
        m8 = np.clip(mf * 255.0 + 0.5, 0, 255).astype(np.uint8)
        b = cv2.bilateralFilter(m8, d=9, sigmaColor=40, sigmaSpace=9)
        refined = b.astype(np.float32) / 255.0

    mr = np.clip(refined * 255.0 + 0.5, 0, 255).astype(np.uint8)
    # 3) Abertura morfológica (remove ruído isolado)
    ksz = max(2, int(os.environ.get("SWITCHX_MASK_OPEN_KERNEL", "2")))
    kernel = np.ones((ksz, ksz), np.uint8)
    mr = cv2.morphologyEx(mr, cv2.MORPH_OPEN, kernel)
    return np.clip(mr.astype(np.float32) / 255.0, 0.0, 1.0)
