from spectral_film_lut.grain_generation import *

# --- Parameters ---
size = 512
scale_factor = 4
grain_count = 1000000

grain_size_mm = 0.01
image_size_mm = 1
image_pixels = 512  # 512 px image
sigma = 0.4


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
    grain_size_px = grain_size_mm * (image_pixels / image_size_mm)
    print(f"Average grain size in pixels: {grain_size_px:.2f}")

    # Log-normal parameters for radius distribution
    meanlog = np.log(grain_size_px * scale_factor)

    size_scaled = size * scale_factor

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
    img = cv2.resize(img.astype(np.uint16), (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)

    grain_size_crop = math.floor(grain_size_px)
    img = img[grain_size_crop:-grain_size_crop, grain_size_crop:-grain_size_crop]

    img = (img - img.mean()) / (img.std() * 10) + 0.5

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

    pixel_size_mm = image_size_mm / image_pixels
    freq_um = np.arange(len(radial_ps)) / (image_pixels * pixel_size_mm)

    # --- Plot ---
    plt.figure(figsize=(6, 4))
    plt.loglog(freq_um, radial_ps)
    plt.xlabel("Spatial frequency (cycles/mm)")
    plt.ylabel("Power")
    plt.title("Radially averaged power spectrum")
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    start = time.time()
    img = compute_grain()
    print(f"Generation time: {time.time() - start:.2f}s")

    compute_power_spectrum(img)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    start = time.time()
    real_size = img.shape[0]
    scale = real_size / image_size_mm
    print(f"{scale=}")
    mu = np.log(grain_size_mm) - 0.5 * sigma ** 2
    x_low = np.exp(mu - sigma)
    x_high = np.exp(mu + sigma)
    img = generate_grain((real_size, real_size), scale, dye_size1_mm=x_low, dye_size2_mm=x_high)
    img = (img - img.mean()) / (img.std() * 10) + 0.5
    print(f"Generation time: {time.time() - start:.2f}s")

    compute_power_spectrum(img[..., 0])

    cv2.imshow('alt_grain', img)
    cv2.waitKey(0)
