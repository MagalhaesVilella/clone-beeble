"""
Pirâmide Laplaciana (4 níveis): substitui só a banda mais grossa pelo re-light,
mantendo detalhe fino do original.

CLI: python relight_laplace.py merge --before a.png --after b.png --out c.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def laplacian_preserve_texture(
    orig_bgr: np.ndarray,
    relit_bgr: np.ndarray,
    levels: int = 4,
) -> np.ndarray:
    import cv2

    if orig_bgr.shape != relit_bgr.shape:
        relit_bgr = cv2.resize(relit_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]))

    def pyr_down(x: np.ndarray) -> np.ndarray:
        return cv2.pyrDown(x)

    def pyr_up_to(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        up = cv2.pyrUp(x)
        if up.shape[0] != hw[0] or up.shape[1] != hw[1]:
            up = cv2.resize(up, (hw[1], hw[0]))
        return up

    def gaussian_pyr(img: np.ndarray, n: int) -> list[np.ndarray]:
        g = [img.astype(np.float32)]
        for _ in range(n - 1):
            g.append(pyr_down(g[-1]))
        return g

    o = orig_bgr.astype(np.float32)
    r = relit_bgr.astype(np.float32)
    Go = gaussian_pyr(o, levels)
    Gr = gaussian_pyr(r, levels)
    laps: list[np.ndarray] = []
    for i in range(levels - 1):
        up = pyr_up_to(Go[i + 1], Go[i].shape[:2])
        laps.append(Go[i] - up)
    base = Gr[levels - 1]
    recon = base
    for i in range(levels - 2, -1, -1):
        up = pyr_up_to(recon, laps[i].shape[:2])
        recon = laps[i] + up
    return np.clip(recon, 0.0, 255.0).astype(np.uint8)


def laplacian_merge_png_paths(before_path: str, after_path: str, out_path: str, levels: int = 4) -> None:
    import cv2

    o = cv2.imread(before_path, cv2.IMREAD_COLOR)
    a = cv2.imread(after_path, cv2.IMREAD_COLOR)
    if o is None or a is None:
        raise FileNotFoundError("PNG inválido ou inexistente.")
    out = laplacian_preserve_texture(o, a, levels=levels)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, out)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--before", required=True)
    m.add_argument("--after", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--levels", type=int, default=4)
    args = p.parse_args()
    if args.cmd == "merge":
        laplacian_merge_png_paths(args.before, args.after, args.out, levels=args.levels)


if __name__ == "__main__":
    main()
