import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random, time
from colorsys import hsv_to_rgb
from matplotlib.path import Path

# ===============================
# Helper functions
# ===============================
def diverse_palette(n_colors=20, seed=None):
    if seed is not None:
        random.seed(seed)
    palette = []
    for _ in range(n_colors):
        h = random.random()
        s = random.uniform(0.35, 0.65)
        v = random.uniform(0.75, 0.95)
        palette.append(hsv_to_rgb(h, s, v))
    return palette

def gaussian_kernel1d(sigma, truncate=3.0):
    if sigma <= 0:
        return np.array([1.0])
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius+1)
    k = np.exp(-(x**2) / (2*sigma*sigma))
    k /= k.sum()
    return k

def separable_gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    k = gaussian_kernel1d(sigma)
    tmp = np.apply_along_axis(lambda row: np.convolve(row, k, mode='same'), 1, img)
    out = np.apply_along_axis(lambda col: np.convolve(col, k, mode='same'), 0, tmp)
    return out

def rasterize_path_mask(path, xmin, xmax, ymin, ymax, W, H):
    xs = np.linspace(xmin, xmax, W)
    ys = np.linspace(ymin, ymax, H)
    X, Y = np.meshgrid(xs, ys)
    points = np.vstack((X.ravel(), Y.ravel())).T
    contains = path.contains_points(points)
    mask = contains.reshape((H, W))
    return mask, X, Y

def render_blob_shaded(ax, poly_x, poly_y, base_color,
                       depth=0.5,
                       light_dir=(-0.6, 0.8),
                       shadow_offset=(0.02, -0.02),
                       scene_alpha=0.9,
                       resolution_scale=600):
    xmin, xmax = poly_x.min(), poly_x.max()
    ymin, ymax = poly_y.min(), poly_y.max()
    pad_x = (xmax - xmin) * 0.45 + 0.01
    pad_y = (ymax - ymin) * 0.45 + 0.01
    xmin_p, xmax_p = xmin - pad_x, xmax + pad_x
    ymin_p, ymax_p = ymin - pad_y, ymax + pad_y

    box_w = xmax_p - xmin_p
    box_h = ymax_p - ymin_p
    approx_pixels = int(max(80, resolution_scale * max(box_w, box_h)))
    W = H = approx_pixels

    verts = np.column_stack([poly_x, poly_y])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts)-1)
    path = Path(verts, codes)

    mask, X, Y = rasterize_path_mask(path, xmin_p, xmax_p, ymin_p, ymax_p, W, H)
    mask_f = mask.astype(float)

    cx = poly_x.mean()
    cy = poly_y.mean()
    rx = (xmax - xmin) / 2 + 1e-6
    ry = (ymax - ymin) / 2 + 1e-6
    nx = (X - cx) / rx
    ny = (Y - cy) / ry
    dist = np.sqrt(nx**2 + ny**2)
    radial = np.clip(1.0 - dist, 0, 1)

    ld = np.array(light_dir, dtype=float)
    ld = ld / (np.linalg.norm(ld) + 1e-9)
    vx = (X - cx)
    vy = (Y - cy)
    vn = np.sqrt(vx**2 + vy**2) + 1e-9
    vxn = vx / vn
    vyn = vy / vn
    dot = vxn * ld[0] + vyn * ld[1]
    directional = (dot * 0.5 + 0.5)

    spec_center_x = cx - ld[0] * 0.15 * rx
    spec_center_y = cy - ld[1] * 0.15 * ry
    sx = (X - spec_center_x) / (rx * 0.6)
    sy = (Y - spec_center_y) / (ry * 0.6)
    spec_dist = np.sqrt(sx**2 + sy**2)
    spec = np.exp(-(spec_dist**2) * 8.0)

    ambient = 0.35 + 0.15 * radial
    shade = ambient + 0.45 * (radial * 0.6 + directional * 0.4)
    shade = np.clip(shade, 0, 1)

    base = np.array(base_color).reshape((1,1,3))
    img_rgb = base * shade[..., None]
    img_rgb = np.clip(img_rgb + (spec[..., None] * 0.6), 0, 1)

    alpha_mask = mask_f * (0.35 + 0.65 * (0.3 + 0.7 * depth)) * scene_alpha
    alpha_mask = np.clip(alpha_mask, 0, 1)

    rgba = np.zeros((H, W, 4), dtype=float)
    rgba[..., :3] = img_rgb
    rgba[..., 3] = alpha_mask

    shadow_strength = 0.22 * (1.0 - depth)
    shadow_sigma = max(2.0, max(W, H) * 0.015)
    shadow_alpha_map = separable_gaussian_blur(mask_f, shadow_sigma)
    shadow_alpha_map = shadow_alpha_map * shadow_strength

    extent_shadow = (xmin_p + shadow_offset[0],
                     xmax_p + shadow_offset[0],
                     ymin_p + shadow_offset[1],
                     ymax_p + shadow_offset[1])
    shadow_rgba = np.zeros((H, W, 4), dtype=float)
    shadow_rgba[..., 3] = shadow_alpha_map
    ax.imshow(shadow_rgba, origin='lower', extent=extent_shadow, interpolation='bilinear', zorder=0)

    extent_blob = (xmin_p, xmax_p, ymin_p, ymax_p)
    ax.imshow(rgba, origin='lower', extent=extent_blob, interpolation='bilinear', zorder=1)

# ===============================
# Streamlit app interface
# ===============================
st.title("🎨 Generative 3D-like Abstract Poster")
st.markdown("**Made by He Pengwei | Arts & Advanced Big Data Week4**")

# sidebar control panel
seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)
n_blobs = st.sidebar.slider("Number of Blobs", 5, 30, 14)
light_x = st.sidebar.slider("Light X Direction", -1.0, 1.0, -0.6)
light_y = st.sidebar.slider("Light Y Direction", -1.0, 1.0, 0.8)
generate_button = st.sidebar.button("🎨 Generate Poster")

if generate_button:
    random.seed(seed)
    np.random.seed(seed)
    fig, ax = plt.subplots(figsize=(7,10))
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_aspect('equal')
    ax.axis('off')
    grad = np.linspace(0.25, 1.0, 600).reshape(-1,1)
    ax.imshow(grad, extent=[0,1,0,1], origin='lower', cmap='coolwarm', alpha=0.18, zorder=-10)

    palette = diverse_palette(n_blobs, seed=seed)
    blobs = []
    for i in range(n_blobs):
        r = random.uniform(0.10, 0.26)
        wobble = random.uniform(0.08, 0.22)
        cx, cy = random.uniform(0.12, 0.88), random.uniform(0.12, 0.88)
        angles = np.linspace(0, 2*np.pi, 280)
        rr = r * (1 + wobble * (np.random.rand(len(angles)) - 0.5) * 2)
        px = cx + rr * np.cos(angles)
        py = cy + rr * np.sin(angles)
        depth = i / float(max(1, n_blobs-1))
        base_color = np.array(palette[i % len(palette)])
        warm_factor = 0.25 * depth
        cool_factor = 0.25 * (1 - depth)
        base_color = base_color + np.array([warm_factor*0.6 - cool_factor*0.1, 0.0, cool_factor*0.7 - warm_factor*0.05])
        base_color = np.clip(base_color, 0, 1)
        blobs.append(dict(px=px, py=py, depth=depth, color=tuple(base_color)))

    blobs_sorted = sorted(blobs, key=lambda b: b['depth'])
    for b in blobs_sorted:
        render_blob_shaded(ax,
                           np.array(b['px']), np.array(b['py']),
                           b['color'],
                           depth=b['depth'],
                           light_dir=(light_x, light_y),
                           shadow_offset=(0.02 * (0.6 + 0.8 * (1-b['depth'])),
                                          -0.02 * (0.6 + 0.8 * (1-b['depth']))),
                           scene_alpha=1.0,
                           resolution_scale=480)

    ax.text(0.5, 0.03, f"Seed: {seed}", fontsize=10, ha='center', va='bottom',
            transform=ax.transAxes, color='gray')
    st.pyplot(fig)
