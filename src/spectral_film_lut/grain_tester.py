from spectral_film_lut.grain_generation import *

# --- Parameters ---
image_size = 128
image_size_mm = 0.5

# TODO: optimize these
grain_size_mm = 0.006
sigma = 0.2

if __name__ == "__main__":
    # --- Generate grain by first method ---
    pixel_size_mm = image_size_mm / image_size
    # --- Generate grain by second method ---
    start = time.time()
    real_size = image_size
    scale = real_size / image_size_mm
    img2 = generate_grain((real_size, real_size), scale, grain_size_mm=grain_size_mm, grain_sigma=sigma)
    img2 = (img2 - img2.mean()) / (img2.std() * 10) + 0.5

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
    image_path = "grain_8mm_500T.png"  # <<< Replace with your file
    width_mm_real = 4.8  # <<< physical width of your image (mm)
    height_mm_real = width_mm_real * (9 / 16)  # <<< physical height of your image (mm)
    crop_mm = image_size_mm  # physical side length of the crop (same as synthetic)

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
    fig, axes = plt.subplots(2, 2, figsize=(12, 14))

    # 2nd method
    axes[0, 0].imshow(img2[..., 0], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("Generated Grain (Method 2)")
    axes[0, 0].axis('off')
    axes[0, 1].loglog(freq_um2, radial_ps2)
    axes[0, 1].set_title("Power Spectrum 2")
    axes[0, 1].set_xlabel("Spatial frequency (cycles/mm)")
    axes[0, 1].set_ylabel("Power")
    axes[0, 1].grid(True, which='both', ls='--', alpha=0.5)

    # Real image
    axes[1, 0].imshow(img_crop_resized, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f"Real Grain (1mm×1mm center crop)")
    axes[1, 0].axis('off')
    axes[1, 1].loglog(freq_um3, radial_ps3)
    axes[1, 1].set_title("Power Spectrum (Real Image)")
    axes[1, 1].set_xlabel("Spatial frequency (cycles/mm)")
    axes[1, 1].set_ylabel("Power")
    axes[1, 1].grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
