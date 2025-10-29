from spectral_film_lut.grain_generation import *
import scipy.optimize as opt
import numpy as np


class GrainOptimizer:
    def __init__(self, real_radial_ps, image_size, image_size_mm, scale):
        self.real_radial_ps = real_radial_ps
        self.image_size = image_size
        self.image_size_mm = image_size_mm
        self.scale = scale
        self.pixel_size_mm = image_size_mm / image_size
        self.evaluation_count = 0

    def objective_function(self, params):
        """
        Objective function to minimize the difference between generated and real power spectra
        """
        grain_size_mm, sigma = params

        # Add some constraints to avoid invalid values
        if grain_size_mm < 0.001 or grain_size_mm > 0.02 or sigma < 0.05 or sigma > 1.0:
            return 1e10  # Large penalty for invalid parameters

        try:
            # Generate grain with current parameters
            img2 = generate_grain((self.image_size, self.image_size), self.scale,
                                  grain_size_mm=grain_size_mm, grain_sigma=sigma)
            img2 = (img2 - img2.mean()) / (img2.std() * 10) + 0.5

            # Compute power spectrum
            F2 = np.fft.fft2(img2[..., 0])
            F2_shift = np.fft.fftshift(F2)
            PS2 = np.abs(F2_shift) ** 2
            y, x = np.indices(PS2.shape)
            center = np.array(PS2.shape) // 2
            r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
            r = r.astype(np.int32)
            radial_ps2 = np.bincount(r.ravel(), PS2.ravel()) / np.bincount(r.ravel())

            # Use only the relevant frequency range (avoid DC component and very high frequencies)
            min_len = min(len(radial_ps2), len(self.real_radial_ps))

            # Focus on the mid-frequency range where grain information is most relevant
            start_idx = 2  # Skip DC and very low frequencies
            end_idx = min(min_len, len(radial_ps2) // 2)  # Use only first half of spectrum

            # Calculate normalized mean squared error
            mse = np.mean((radial_ps2[start_idx:end_idx] - self.real_radial_ps[start_idx:end_idx]) ** 2)

            self.evaluation_count += 1
            if self.evaluation_count % 10 == 0:
                print(f"Eval {self.evaluation_count}: grain_size={grain_size_mm:.6f}, sigma={sigma:.4f}, MSE={mse:.6e}")

            return mse

        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e10


def optimize_grain_parameters(real_radial_ps, image_size, image_size_mm, scale,
                              initial_guess=None):
    """
    Optimize grain parameters to match real sample power spectrum
    """
    if initial_guess is None:
        initial_guess = [0.006, 0.2]  # grain_size_mm, sigma

    bounds = [(0.001, 0.02), (0.05, 1.0)]  # Reasonable bounds for parameters

    optimizer = GrainOptimizer(real_radial_ps, image_size, image_size_mm, scale)

    print("Starting optimization...")
    print(f"Initial guess: grain_size_mm={initial_guess[0]:.4f}, sigma={initial_guess[1]:.3f}")

    # Try multiple optimization methods
    methods = ['Nelder-Mead', 'Powell', 'COBYLA']

    best_result = None
    best_value = float('inf')

    for method in methods:
        print(f"\n--- Trying {method} method ---")
        optimizer.evaluation_count = 0

        try:
            result = opt.minimize(
                optimizer.objective_function,
                initial_guess,
                method=method,
                bounds=bounds if method != 'Nelder-Mead' and method != 'Powell' else None,
                options={'disp': True, 'maxiter': 100}
            )

            if result.fun < best_value:
                best_value = result.fun
                best_result = result
                print(f"New best result with {method}: MSE={result.fun:.6e}")

        except Exception as e:
            print(f"Method {method} failed: {e}")
            continue

    if best_result is not None:
        print(f"\nBest optimization result:")
        print(f"Method: {best_result.method if hasattr(best_result, 'method') else 'Unknown'}")
        print(f"Parameters: grain_size_mm={best_result.x[0]:.6f}, sigma={best_result.x[1]:.4f}")
        print(f"MSE: {best_result.fun:.6e}")
        print(f"Success: {best_result.success}")
        if hasattr(best_result, 'message'):
            print(f"Message: {best_result.message}")

        return best_result.x, best_result.fun
    else:
        print("All optimization methods failed. Using initial guess.")
        return initial_guess, optimizer.objective_function(initial_guess)


# --- Parameters ---
image_size = 256
image_size_mm = 3

if __name__ == "__main__":
    # --- Load and process real grain image first ---
    image_path = "grain_35mm_200T.png"  # <<< Replace with your file
    width_mm_real = 24  # <<< physical width of your image (mm)
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

    # --- Compute crop size in pixels for target size ---
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
    pixel_size_mm = image_size_mm / image_size
    F3 = np.fft.fft2(img_crop_resized)
    F3_shift = np.fft.fftshift(F3)
    PS3 = np.abs(F3_shift) ** 2
    y, x = np.indices(PS3.shape)
    center = np.array(PS3.shape) // 2
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int32)
    radial_ps3 = np.bincount(r.ravel(), PS3.ravel()) / np.bincount(r.ravel())
    freq_um3 = np.arange(len(radial_ps3)) / (image_size * pixel_size_mm)

    # --- Also try a grid search first to find good initial parameters ---
    print("Performing initial grid search...")
    scale = image_size / image_size_mm
    best_grid_params = None
    best_grid_mse = float('inf')

    # Test a range of reasonable parameters
    grain_sizes = [0.003, 0.005, 0.007, 0.009, 0.012]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.6]

    temp_optimizer = GrainOptimizer(radial_ps3, image_size, image_size_mm, scale)

    for gs in grain_sizes:
        for s in sigmas:
            mse = temp_optimizer.objective_function([gs, s])
            if mse < best_grid_mse:
                best_grid_mse = mse
                best_grid_params = [gs, s]
            print(f"Grid: grain_size={gs:.4f}, sigma={s:.3f}, MSE={mse:.6e}")

    print(
        f"\nBest from grid search: grain_size={best_grid_params[0]:.6f}, sigma={best_grid_params[1]:.4f}, MSE={best_grid_mse:.6e}")

    # --- Optimize parameters using best from grid search as initial guess ---
    optimal_params, final_mse = optimize_grain_parameters(
        radial_ps3, image_size, image_size_mm, scale, initial_guess=best_grid_params
    )

    grain_size_mm_opt, sigma_opt = optimal_params

    # --- Generate final optimized grain ---
    print("\nGenerating final optimized grain...")
    img_optimized = generate_grain((image_size, image_size), scale,
                                   grain_size_mm=grain_size_mm_opt, grain_sigma=sigma_opt)
    img_optimized = (img_optimized - img_optimized.mean()) / (img_optimized.std() * 10) + 0.5

    # --- Power spectrum for optimized grain ---
    F_opt = np.fft.fft2(img_optimized[..., 0])
    F_opt_shift = np.fft.fftshift(F_opt)
    PS_opt = np.abs(F_opt_shift) ** 2
    y, x = np.indices(PS_opt.shape)
    center = np.array(PS_opt.shape) // 2
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int32)
    radial_ps_opt = np.bincount(r.ravel(), PS_opt.ravel()) / np.bincount(r.ravel())
    freq_um_opt = np.arange(len(radial_ps_opt)) / (image_size * pixel_size_mm)

    # --- Plot comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Optimized grain
    axes[0, 0].imshow(img_optimized[..., 0], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f"Optimized Grain\nsize={grain_size_mm_opt:.4f}mm, σ={sigma_opt:.3f}")
    axes[0, 0].axis('off')

    # Real image
    axes[1, 0].imshow(img_crop_resized, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f"Real Grain ({image_size_mm}mm×{image_size_mm}mm crop)")
    axes[1, 0].axis('off')

    # Power spectra comparison
    axes[0, 1].loglog(freq_um_opt, radial_ps_opt, 'b-', label='Optimized', alpha=0.7)
    axes[0, 1].set_title("Power Spectrum (Optimized)")
    axes[0, 1].set_xlabel("Spatial frequency (cycles/mm)")
    axes[0, 1].set_ylabel("Power")
    axes[0, 1].grid(True, which='both', ls='--', alpha=0.5)
    axes[0, 1].legend()

    axes[1, 1].loglog(freq_um3, radial_ps3, 'r-', label='Real', alpha=0.7)
    axes[1, 1].set_title("Power Spectrum (Real Image)")
    axes[1, 1].set_xlabel("Spatial frequency (cycles/mm)")
    axes[1, 1].set_ylabel("Power")
    axes[1, 1].grid(True, which='both', ls='--', alpha=0.5)
    axes[1, 1].legend()

    # Combined power spectra for direct comparison
    axes[0, 2].loglog(freq_um_opt, radial_ps_opt, 'b-', label='Optimized', alpha=0.7)
    axes[0, 2].loglog(freq_um3, radial_ps3, 'r-', label='Real', alpha=0.7)
    axes[0, 2].set_title("Power Spectrum Comparison")
    axes[0, 2].set_xlabel("Spatial frequency (cycles/mm)")
    axes[0, 2].set_ylabel("Power")
    axes[0, 2].grid(True, which='both', ls='--', alpha=0.5)
    axes[0, 2].legend()

    # Difference in power spectra
    min_len = min(len(radial_ps_opt), len(radial_ps3))
    ps_diff = radial_ps_opt[:min_len] - radial_ps3[:min_len]
    axes[1, 2].semilogx(freq_um_opt[:min_len], ps_diff, 'g-', alpha=0.7)
    axes[1, 2].set_title("Power Spectrum Difference\n(Optimized - Real)")
    axes[1, 2].set_xlabel("Spatial frequency (cycles/mm)")
    axes[1, 2].set_ylabel("Power Difference")
    axes[1, 2].grid(True, which='both', ls='--', alpha=0.5)
    axes[1, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n=== FINAL OPTIMIZATION RESULTS ===")
    print(f"Optimal grain_size_mm: {grain_size_mm_opt:.6f}")
    print(f"Optimal sigma: {sigma_opt:.4f}")
    print(f"Final MSE: {final_mse:.6e}")