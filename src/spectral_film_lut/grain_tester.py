import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from spectral_film_lut.grain_generation import *

# --- Parameters ---
image_size = 128
scale_factor = 4
grain_count = 10000

grain_size_mm = 0.006
image_size_mm = 0.5
sigma = 0.1


@njit(parallel=True, fastmath=True)
def add_grains_numba(centers, radii_indices, offset_starts, offset_lengths, all_offsets_x, all_offsets_y, size_scaled):
    img = np.zeros((size_scaled, size_scaled), dtype=np.uint32)
    for i in prange(centers.shape[0]):
        cx, cy = centers[i]
        ridx = radii_indices[i]

        start_i = offset_starts[ridx]
        end_i = start_i + offset_lengths[ridx]

        for k in range(start_i, end_i):
            xi = cx + all_offsets_x[k]
            yi = cy + all_offsets_y[k]
            if 0 <= xi < size_scaled and 0 <= yi < size_scaled:
                img[xi, yi] += 1
    return img


def compute_grain():
    grain_size_px = grain_size_mm * (image_size / image_size_mm)
    print(f"Average grain size in pixels: {grain_size_px:.2f}")

    # Log-normal parameters for radius distribution
    meanlog = np.log(grain_size_px * scale_factor)

    size_scaled = image_size * scale_factor

    # --- Generate random grain centers and radii ---
    rng = np.random.default_rng()
    centers = rng.integers(0, size_scaled, (grain_count, 2), dtype=np.int32)

    # Draw radii from log-normal distribution
    radii = rng.lognormal(mean=meanlog, sigma=sigma, size=grain_count)
    radii = np.clip(radii.astype(np.int32), 1, 8 * scale_factor)  # limit extremes

    # --- Precompute masks for each unique radius ---
    unique_radii = np.unique(radii)
    mask_offsets = {}
    for r in unique_radii:
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        mask = (xx ** 2 + yy ** 2) <= r ** 2
        ys, xs = np.nonzero(mask)
        mask_offsets[r] = (xs - r, ys - r)  # relative offsets

    # --- Convert to arrays for Numba ---
    # Build arrays for fast access: offsets concatenated
    radius_to_index = {}  # radius -> start index
    all_offsets_x, all_offsets_y = [], []
    start_idx = 0
    for r in unique_radii:
        xs, ys = mask_offsets[r]
        all_offsets_x.extend(xs)
        all_offsets_y.extend(ys)
        radius_to_index[r] = start_idx
        start_idx += len(xs)
    all_offsets_x = np.array(all_offsets_x, dtype=np.int32)
    all_offsets_y = np.array(all_offsets_y, dtype=np.int32)

    # Start indices and lengths
    offset_starts = np.zeros(len(unique_radii), dtype=np.int32)
    offset_lengths = np.zeros(len(unique_radii), dtype=np.int32)
    for i, r in enumerate(unique_radii):
        offset_starts[i] = radius_to_index[r]
        offset_lengths[i] = len(mask_offsets[r][0])

    # Map radii to their index in unique_radii array
    radii_map = {r: i for i, r in enumerate(unique_radii)}
    radii_indices = np.array([radii_map[r] for r in radii], dtype=np.int32)

    # --- Run ---
    img = add_grains_numba(centers, radii_indices, offset_starts, offset_lengths, all_offsets_x, all_offsets_y,
                           size_scaled)

    # --- Downscale and normalize ---
    img = cv2.resize(img.astype(np.uint16), (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)

    grain_size_crop = math.floor(grain_size_px * 2)
    img_cropped = img[grain_size_crop:-grain_size_crop, grain_size_crop:-grain_size_crop]

    img = (img - img_cropped.mean()) / (img_cropped.std() * 10) + 0.5

    return img


def compute_power_spectrum(img):
    # --- Compute 2D Fourier Transform ---
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)  # center zero frequency
    PS = np.abs(F_shift) ** 2  # power spectrum

    # --- Radially average the power spectrum ---
    y, x = np.indices(PS.shape)
    center = np.array(PS.shape) // 2
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    # Bin the radial distances
    r = r.astype(np.int32)
    radial_ps = np.bincount(r.ravel(), PS.ravel()) / np.bincount(r.ravel())

    pixel_size_mm = image_size_mm / image_size
    freq_um = np.arange(len(radial_ps)) / (image_size * pixel_size_mm)

    # --- Plot ---
    plt.figure(figsize=(6, 4))
    plt.loglog(freq_um, radial_ps)
    plt.xlabel("Spatial frequency (cycles/mm)")
    plt.ylabel("Power")
    plt.title("Radially averaged power spectrum")
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.show()


simulate_algorithmically = False


if __name__ == "__main__":
    # --- Generate grain by first method ---
    start = time.time()
    pixel_size_mm = image_size_mm / image_size
    if simulate_algorithmically:
        img1 = compute_grain()
        print(f"Generation time (method 1): {time.time() - start:.2f}s")

        # --- Power spectrum for img1 ---
        F1 = np.fft.fft2(img1)
        F1_shift = np.fft.fftshift(F1)
        PS1 = np.abs(F1_shift) ** 2
        y, x = np.indices(PS1.shape)
        center = np.array(PS1.shape) // 2
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r = r.astype(np.int32)
        radial_ps1 = np.bincount(r.ravel(), PS1.ravel()) / np.bincount(r.ravel())
        freq_um1 = np.arange(len(radial_ps1)) / (image_size * pixel_size_mm)

    # --- Generate grain by second method ---
    start = time.time()
    if simulate_algorithmically:
        real_size = img1.shape[0]
    else:
        real_size = image_size
    scale = real_size / image_size_mm
    mu = np.log(grain_size_mm) - 0.5 * sigma ** 2
    x_low = np.exp(mu - sigma)
    x_high = np.exp(mu + sigma)
    print(x_low, x_high)
    img2 = generate_grain((real_size, real_size), scale, dye_size1_mm=x_low, dye_size2_mm=x_high)
    img2 = (img2 - img2.mean()) / (img2.std() * 10) + 0.5
    print(f"Generation time (method 2): {time.time() - start:.2f}s")

    # --- Power spectrum for img2 ---
    F2 = np.fft.fft2(img2[..., 0])
    F2_shift = np.fft.fftshift(F2)
    PS2 = np.abs(F2_shift) ** 2
    y, x = np.indices(PS2.shape)
    center = np.array(PS2.shape) // 2
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int32)
    radial_ps2 = np.bincount(r.ravel(), PS2.ravel()) / np.bincount(r.ravel())
    freq_um2 = np.arange(len(radial_ps2)) / (image_size * pixel_size_mm)

    # --- Load real grain image ---
    image_path = "grain_35mm_200T.png"  # <<< Replace with your file
    width_mm_real = 24            # <<< physical width of your image (mm)
    height_mm_real = width_mm_real * (9 / 16)   # <<< physical height of your image (mm)
    crop_mm = image_size_mm         # physical side length of the crop (same as synthetic)

    img_real = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_real is None:
        raise ValueError(f"Could not read image from {image_path}")

    img_real = img_real.astype(np.float32) / 255.0

    # --- Extract green channel if RGB ---
    if img_real.ndim == 3:
        img_real = img_real[..., 1]

    # --- Compute pixels per millimeter ---
    h, w = img_real.shape
    px_per_mm_x = w / width_mm_real
    px_per_mm_y = h / height_mm_real
    px_per_mm = (px_per_mm_x + px_per_mm_y) / 2.0  # average if slightly different

    # --- Compute crop size in pixels for 1mm × 1mm ---
    crop_px = int(round(px_per_mm * crop_mm))

    # --- Center crop in physical space ---
    cy, cx = h // 2, w // 2
    half = crop_px // 2
    img_crop = img_real[cy - half:cy + half, cx - half:cx + half]

    if img_crop.shape[0] != crop_px or img_crop.shape[1] != crop_px:
        raise ValueError("Crop region extends beyond image bounds. Adjust dimensions or crop size.")

    # --- Rescale to same resolution as synthetic sample ---
    img_crop_resized = cv2.resize(img_crop, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # --- Normalize the same way ---
    img_crop_resized = (img_crop_resized - img_crop_resized.mean()) / (img_crop_resized.std() * 10) + 0.5

    # --- Power spectrum for real grain crop ---
    F3 = np.fft.fft2(img_crop_resized)
    F3_shift = np.fft.fftshift(F3)
    PS3 = np.abs(F3_shift) ** 2
    y, x = np.indices(PS3.shape)
    center = np.array(PS3.shape) // 2
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int32)
    radial_ps3 = np.bincount(r.ravel(), PS3.ravel()) / np.bincount(r.ravel())
    freq_um3 = np.arange(len(radial_ps3)) / (image_size * pixel_size_mm)

    # --- Plot all three ---
    fig, axes = plt.subplots(3 if simulate_algorithmically else 2, 2, figsize=(12, 14))

    i = 0
    if simulate_algorithmically:
        # 1st method
        axes[i, 0].imshow(img1, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title("Generated Grain (Method 1)")
        axes[i, 0].axis('off')
        axes[i, 1].loglog(freq_um1, radial_ps1)
        axes[i, 1].set_title("Power Spectrum 1")
        axes[i, 1].set_xlabel("Spatial frequency (cycles/mm)")
        axes[i, 1].set_ylabel("Power")
        axes[i, 1].grid(True, which='both', ls='--', alpha=0.5)
        i += 1

    # 2nd method
    axes[i, 0].imshow(img2[..., 0], cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_title("Generated Grain (Method 2)")
    axes[i, 0].axis('off')
    axes[i, 1].loglog(freq_um2, radial_ps2)
    axes[i, 1].set_title("Power Spectrum 2")
    axes[i, 1].set_xlabel("Spatial frequency (cycles/mm)")
    axes[i, 1].set_ylabel("Power")
    axes[i, 1].grid(True, which='both', ls='--', alpha=0.5)
    i += 1

    # Real image
    axes[i, 0].imshow(img_crop_resized, cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_title(f"Real Grain (1mm×1mm center crop)")
    axes[i, 0].axis('off')
    axes[i, 1].loglog(freq_um3, radial_ps3)
    axes[i, 1].set_title("Power Spectrum (Real Image)")
    axes[i, 1].set_xlabel("Spatial frequency (cycles/mm)")
    axes[i, 1].set_ylabel("Power")
    axes[i, 1].grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
