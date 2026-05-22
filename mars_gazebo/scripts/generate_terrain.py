#!/usr/bin/env python3
"""
Generate Mars rough terrain heightmap for MuJoCo.

Produces a 16-bit PNG heightmap matching the mjlab training terrain:
  - Random rough (dominant): noise_range 0.02-0.15m, noise_step 0.04m
  - Perlin noise: smooth undulations up to 0.6m (scaled down for single patch)
  - Gentle slopes: simulating crater walls

MuJoCo hfield maps pixel values [0, 65535] to [0, max_height].
The hfield size attribute controls x/y extent and max z height.

Usage:
  python3 generate_terrain.py
  python3 generate_terrain.py --output /path/to/terrain.png --size 200 --roughness 0.12
"""
import argparse
import numpy as np

def perlin_noise_2d(shape, scale=10.0, octaves=4, persistence=0.5, seed=42):
    """Generate 2D Perlin-like noise using numpy (no external deps)."""
    rng = np.random.default_rng(seed)
    noise = np.zeros(shape)
    amplitude = 1.0
    frequency = 1.0

    for _ in range(octaves):
        # Generate random gradients at grid points
        grid_h = max(2, int(shape[0] / scale * frequency) + 1)
        grid_w = max(2, int(shape[1] / scale * frequency) + 1)

        # Random gradient angles
        angles = rng.uniform(0, 2 * np.pi, (grid_h, grid_w))
        grad_x = np.cos(angles)
        grad_y = np.sin(angles)

        # Coordinate grids
        y_coords = np.linspace(0, grid_h - 1, shape[0])
        x_coords = np.linspace(0, grid_w - 1, shape[1])
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Integer and fractional parts
        x0 = np.floor(xx).astype(int)
        y0 = np.floor(yy).astype(int)
        x1 = np.minimum(x0 + 1, grid_w - 1)
        y1 = np.minimum(y0 + 1, grid_h - 1)

        # Fractional position within cell
        fx = xx - x0
        fy = yy - y0

        # Smoothstep interpolation weights
        sx = fx * fx * (3 - 2 * fx)
        sy = fy * fy * (3 - 2 * fy)

        # Dot products with gradients at corners
        def dot_grad(gx, gy, ix, iy):
            return gx[iy, ix] * (xx - ix) + gy[iy, ix] * (yy - iy)

        n00 = dot_grad(grad_x, grad_y, x0, y0)
        n10 = dot_grad(grad_x, grad_y, x1, y0)
        n01 = dot_grad(grad_x, grad_y, x0, y1)
        n11 = dot_grad(grad_x, grad_y, x1, y1)

        # Bilinear interpolation
        nx0 = n00 * (1 - sx) + n10 * sx
        nx1 = n01 * (1 - sx) + n11 * sx
        layer = nx0 * (1 - sy) + nx1 * sy

        noise += amplitude * layer
        amplitude *= persistence
        frequency *= 2.0

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
    return noise


def random_rough(shape, noise_range=(0.02, 0.15), noise_step=0.04, seed=42):
    """Generate random rough terrain matching mjlab's random_rough config.

    Each cell gets a quantized random height in noise_range.
    noise_step controls the quantization (coarser = more blocky).
    """
    rng = np.random.default_rng(seed)
    # Generate continuous random noise
    raw = rng.uniform(noise_range[0], noise_range[1], shape)
    # Quantize to noise_step
    if noise_step > 0:
        raw = np.round(raw / noise_step) * noise_step
    # Normalize to [0, 1]
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-10)
    return raw


def gentle_slope(shape, max_angle_deg=15):
    """Generate a gentle slope (simulating crater wall approach)."""
    y = np.linspace(0, 1, shape[0])[:, None]
    x = np.linspace(0, 1, shape[1])[None, :]
    # Radial slope from center
    cx, cy = 0.5, 0.5
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    slope = dist / dist.max()
    return slope


def generate_mars_terrain(size=200, roughness=0.12, seed=42):
    """Generate composite Mars terrain heightmap.

    Blends:
      - 50% random rough (regolith, small rocks)
      - 30% Perlin noise (organic undulations)
      - 20% gentle slopes (crater approaches)

    Returns heightmap as float32 array in [0, 1].
    """
    shape = (size, size)

    # Random rough (dominant, like training)
    rough = random_rough(shape, noise_range=(0.02, roughness),
                         noise_step=0.04, seed=seed)

    # Perlin noise (smooth undulations)
    perlin = perlin_noise_2d(shape, scale=8.0, octaves=4,
                             persistence=0.5, seed=seed + 1)

    # Gentle slope component
    slope = gentle_slope(shape, max_angle_deg=12)

    # Composite blend
    terrain = 0.50 * rough + 0.30 * perlin + 0.20 * slope

    # Normalize final result to [0, 1]
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-10)

    # Add a flat landing zone around the robot spawn (center, ~1m radius)
    center = size // 2
    radius_px = int(size * 0.1)  # 10% of terrain = ~1m at 10m terrain
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)

    # Smooth blend to flat at center (spawn point)
    blend = np.clip(dist_from_center / radius_px, 0, 1)
    # Smoothstep
    blend = blend * blend * (3 - 2 * blend)
    spawn_height = terrain[center, center]
    terrain = spawn_height * (1 - blend) + terrain * blend

    return terrain


def save_heightmap_png(heightmap, filepath):
    """Save heightmap as 16-bit PNG for MuJoCo hfield.

    MuJoCo reads PNG heightmaps where pixel value maps to elevation:
      0 = minimum height (0)
      65535 = maximum height (hfield size[2])
    """
    try:
        from PIL import Image
    except ImportError:
        # Fallback: save as raw numpy, user can convert
        npy_path = filepath.replace('.png', '.npy')
        np.save(npy_path, heightmap)
        print(f"PIL not available. Saved as numpy: {npy_path}")
        print("Install Pillow: pip install Pillow")
        return npy_path

    # Convert [0,1] float to 16-bit unsigned int
    h16 = (heightmap * 65535).astype(np.uint16)
    img = Image.fromarray(h16, mode='I;16')
    img.save(filepath)
    print(f"Saved {filepath}: {heightmap.shape[0]}x{heightmap.shape[1]}, "
          f"height range [{heightmap.min():.3f}, {heightmap.max():.3f}]")
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Generate Mars terrain heightmap')
    parser.add_argument('--output', '-o',
        default='/home/sid/projects25/src/HANUMAN/mars_gazebo/unitree_g1_mjcf/mars_terrain.png',
        help='Output PNG path')
    parser.add_argument('--size', type=int, default=200,
        help='Heightmap resolution (NxN pixels)')
    parser.add_argument('--roughness', type=float, default=0.12,
        help='Max roughness in meters (matches noise_range upper bound)')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for reproducibility')
    args = parser.parse_args()

    print(f"Generating {args.size}x{args.size} Mars terrain (roughness={args.roughness}m)...")
    terrain = generate_mars_terrain(
        size=args.size, roughness=args.roughness, seed=args.seed)

    filepath = save_heightmap_png(terrain, args.output)

    # Print stats
    print(f"\nTerrain statistics:")
    print(f"  Min height: {terrain.min():.4f}")
    print(f"  Max height: {terrain.max():.4f}")
    print(f"  Mean height: {terrain.mean():.4f}")
    print(f"  Std dev: {terrain.std():.4f}")
    print(f"\nTo use in MuJoCo scene_ros2.xml:")
    print(f'  <hfield name="mars_terrain" file="{filepath}"')
    print(f'    size="5 5 0.15 0.01"/>')


if __name__ == '__main__':
    main()