import cv2
import numpy as np
import os

# =========================
# Time-of-day brightness presets (simulation only)
# =========================
TIME_BRIGHTNESS = {
    "morning": 1.00,  # baseline brightness
    "noon":    1.25,  # brighter at midday
    "evening": 0.75,  # dimmer at dusk
}

# =========================
# Altitude blur presets (simulation only)
# =========================
ALTITUDE_KERNEL = {
    "low":    3,   # minimal blur (low altitude)
    "medium": 7,   # moderate blur (medium altitude)
    "high":   11,  # strong blur (high altitude)
}

def adjust_brightness(img, factor=1.0):
    """Multiply pixel intensities by a constant factor."""
    out = img.astype(np.float32) * float(factor)
    return np.clip(out, 0, 255).astype(np.uint8)

def adjust_brightness_by_time(img, time_of_day):
    """Apply brightness preset for a given simulated time of day."""
    if time_of_day not in TIME_BRIGHTNESS:
        raise ValueError(f"Unknown time_of_day '{time_of_day}'. Use one of {list(TIME_BRIGHTNESS.keys())}.")
    return adjust_brightness(img, TIME_BRIGHTNESS[time_of_day])

def simulate_blur_by_altitude(img, altitude):
    """Apply Gaussian blur for a given simulated flight altitude."""
    if altitude not in ALTITUDE_KERNEL:
        raise ValueError(f"Unknown altitude '{altitude}'. Use one of {list(ALTITUDE_KERNEL.keys())}.")
    k = ALTITUDE_KERNEL[altitude]
    if k % 2 == 0:
        k += 1  # ensure odd kernel size
    return cv2.GaussianBlur(img, (k, k), 0)

# =========================
# Main
# =========================
# Input and output (edit these paths as needed)
image_path = (r"D:/Second Life Internship/Albania/Albania Images stitched -20250617T131807Z-1-002/Albania Images "
              r"stitched/Mission 1.1_compressed/DJI_20250502123531_0013_D_stitch.webp")
output_dir = (r"D:/Second Life Internship/Albania/Albania Images stitched -20250617T131807Z-1-002/Albania Images "
              r"stitched/Mission 1.1_compressed")

os.makedirs(output_dir, exist_ok=True)

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

base_name = os.path.splitext(os.path.basename(image_path))[0]

# Save original once (optional)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.webp"), img)

# Generate 3 (time) × 3 (altitude) simulated variants
times = ["morning", "noon", "evening"]
alts  = ["low", "medium", "high"]

for t in times:
    # First apply time-of-day brightness
    t_img = adjust_brightness_by_time(img, t)
    # Then apply three altitude levels
    for a in alts:
        out = simulate_blur_by_altitude(t_img, a)
        out_path = os.path.join(output_dir, f"{base_name}_sim_time-{t}_alt-{a}.webp")
        cv2.imwrite(out_path, out)
        print(f"Saved: {out_path}")

print("Done. All outputs are purely simulated (no real EXIF/metadata used).")
