"""Radial D4σ beam width evaluation with Hofer circular-aperture correction (Eq. 23, Opt. Eng. 56(4), 2017)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def radial_profile(image: np.ndarray, bins: int = 1200):
    """Compute azimuthally averaged radial intensity profiles within the largest inscribed circle."""
    h, w = image.shape
    border = np.concatenate([image[:30].ravel(), image[-30:].ravel(), image[:, :30].ravel(), image[:, -30:].ravel()])
    bg = float(np.median(border))
    signal = image.astype(np.float64) - bg
    pos = np.clip(signal, 0.0, None)

    y, x = np.ogrid[:h, :w]
    flux = float(pos.sum())
    if flux <= 0:
        raise ValueError("No positive signal above background.")
    cx, cy = float((pos * x).sum() / flux), float((pos * y).sum() / flux)

    r_max = min(cx, cy, w - cx, h - cy)
    edges = np.linspace(0.0, r_max, bins + 1)
    radii = 0.5 * (edges[:-1] + edges[1:])

    radius = np.hypot(x - cx, y - cy)
    mask = radius < r_max
    idx = np.digitize(radius[mask], edges) - 1
    counts = np.bincount(idx, minlength=bins)
    divisor = np.maximum(counts, 1)

    sat_val = np.iinfo(image.dtype).max
    sat_frac = np.bincount(idx, image[mask] == sat_val, minlength=bins) / divisor
    if np.median(sat_frac[:max(5, bins // 20)]) >= 0.20:
        raise ValueError("Central saturation detected.")

    prof_pos = np.bincount(idx, pos[mask], minlength=bins) / divisor
    prof_signed = np.bincount(idx, signal[mask], minlength=bins) / divisor
    return radii, edges, prof_pos, prof_signed


def diameter(radii: np.ndarray, edges: np.ndarray, profile: np.ndarray, limit: float) -> float:
    """Radial D4σ = 4 * sqrt(sum(W * r^2) / (2 * sum(W))) integrated up to `limit`."""
    m = radii <= limit
    weights = (profile * 2.0 * np.pi * radii * np.diff(edges))[m]
    power = weights.sum()
    if power <= 0:
        return np.nan
    variance = (weights * radii[m] ** 2).sum() / (2.0 * power)
    return float(4.0 * np.sqrt(variance)) if variance > 0 else np.nan


def hofer_gaussian_factor(aperture_radius: float, e2_radius: float) -> float:
    """Gaussian circular-aperture correction factor (Hofer et al., 2017, Eq. 23)."""
    t = 2.0 * (aperture_radius / e2_radius) ** 2
    beta = -np.expm1(-t)
    return float((1.0 - t * np.exp(-t) / beta) ** (-0.5))


def measure(image: np.ndarray, bins: int = 1200) -> dict[str, float]:
    """Evaluate 1/e²-limited, inscribed-aperture, and Hofer-corrected D4σ diameters."""
    radii, edges, positive, signed = radial_profile(image, bins)
    normalized = positive / positive[:40].mean()
    j = int(np.flatnonzero(normalized < np.exp(-2.0))[0])
    radius_e2 = float(np.interp(np.exp(-2.0), normalized[j - 1:j + 1][::-1], radii[j - 1:j + 1][::-1]))

    inscribed_radius = float(edges[-1])
    d4_at_1e2 = diameter(radii, edges, signed, radius_e2)
    d4_inscribed = diameter(radii, edges, signed, inscribed_radius)
    d4_hofer = d4_inscribed * hofer_gaussian_factor(inscribed_radius, radius_e2)
    return dict(
        r_e2_px=radius_e2,
        inscribed_radius_px=inscribed_radius,
        aperture_over_e2=float(inscribed_radius / radius_e2),
        d4_at_1e2_px=d4_at_1e2,
        d4_inscribed_px=d4_inscribed,
        d4_hofer_px=d4_hofer,
        d4_px=d4_hofer,
    )


IMAGE_EXTS = {".bmp", ".png", ".tif", ".tiff"}


def main():
    """Run batch D4σ evaluation and print summary statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="*", help="Grayscale image files.")
    parser.add_argument("--image-dir", type=Path, help="Directory of grayscale images (BMP, PNG, TIFF).")
    parser.add_argument("--scale-um-per-pixel", type=float, help="Calibrated scale (µm/px).")
    parser.add_argument("--scale-half-width", type=float, help="Scale uncertainty half-width (µm/px).")
    parser.add_argument("--csv", type=Path, help="Optional output CSV path.")
    args = parser.parse_args()

    if args.image_dir and args.images:
        parser.error("Provide either image files or --image-dir, not both.")

    use_bundled = not args.images and args.image_dir is None
    if use_bundled:
        script_dir = Path(__file__).resolve().parent
        img_dir = script_dir / "synthetic_demo_images"
        image_paths = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not image_paths:
            image_paths = sorted(script_dir.glob("synthetic_0[123].png"))
        scale = 0.057 if args.scale_um_per_pixel is None else args.scale_um_per_pixel
        half_width = 0.001 if args.scale_half_width is None else args.scale_half_width
    elif args.image_dir is not None:
        image_paths = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        scale = args.scale_um_per_pixel
        half_width = args.scale_half_width
    else:
        image_paths = list(args.images)
        scale = args.scale_um_per_pixel
        half_width = args.scale_half_width

    if not image_paths:
        parser.error("No input images found.")
    if scale is None or scale <= 0:
        parser.error("Provide a positive --scale-um-per-pixel.")

    rows = []
    for idx, path in enumerate(image_paths, start=1):
        img = np.array(Image.open(path))
        res = measure(img)
        rows.append(dict(image=idx, filename=path.name, **res, d4_um=res["d4_px"] * scale))

    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        print(f"Image {row['image']} ({row['filename']}): {row['d4_px']:.2f} px, {row['d4_um']:.3f} µm "
              f"(1/e²-limited {row['d4_at_1e2_px'] * scale:.2f} µm; inscribed {row['d4_inscribed_px'] * scale:.2f}, "
              f"Hofer {row['d4_hofer_px'] * scale:.2f} µm; R_ap/w={row['aperture_over_e2']:.3f})")

    mean_e2_px = float(np.mean([2.0 * r["r_e2_px"] for r in rows]))
    mean_lim_px = float(np.mean([r["d4_at_1e2_px"] for r in rows]))
    mean_insc_px = float(np.mean([r["d4_inscribed_px"] for r in rows]))
    mean_px = float(np.mean([r["d4_px"] for r in rows]))
    print(f"Processed {len(rows)} images.")
    print(f"Mean 1/e² diameter: {mean_e2_px:.2f} px, {scale * mean_e2_px:.3f} µm at {scale:g} µm/px.")
    print(f"Mean 1/e²-limited D4σ: {mean_lim_px:.2f} px, {scale * mean_lim_px:.3f} µm.")
    print(f"Mean aperture-integrated D4σ: {mean_insc_px:.2f} px, {scale * mean_insc_px:.3f} µm.")
    print(f"Final mean Hofer-corrected D4σ: {mean_px:.2f} px, {scale * mean_px:.3f} µm at {scale:g} µm/px.")

    if len(rows) >= 2:
        sd = float(np.std([r["d4_um"] for r in rows], ddof=1))
        u_a = sd / np.sqrt(len(rows))
        u_cal = float(mean_px * half_width / np.sqrt(3.0)) if half_width is not None else 0.0
        u_c = float(np.hypot(u_a, u_cal))
        print(f"Repeatability u_A: {u_a:.3f} µm (SD {sd:.3f} µm); calibration u_cal: {u_cal:.3f} µm.")
        print(f"Partial combined u_c: {u_c:.3f} µm; expanded U (k=2): {2.0 * u_c:.3f} µm.")


if __name__ == "__main__":
    main()
