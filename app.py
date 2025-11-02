import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random, io, time
from colorsys import hsv_to_rgb
from matplotlib.path import Path

# ===============================
# 🎨 Palette & Utility
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
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2 * sigma * sigma))
    k /= k.sum()
    return k

def separable_gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    k = gaussian_kernel1d(sigma)
    tmp = np.apply_along_axis(lambda row: np.convolve(row, k, mode="same"), 1, img)
    out = np.apply_along_axis(lambda col: np.convolve(col, k, mode="same"), 0, tmp)
    return out

def rasterize_path_mask(path, xmin, xmax, ymin, ymax, W, H):
    xs = np.linspace(xmin, xmax, W)
    ys = np.linspace(ymin, ymax, H)
    X, Y = np.meshgrid(xs, ys)
    points = np.vstack((X.ravel(), Y.ravel())).T
    contains = path.contains_points(points)
    mask = contains.reshape((H, W))
    return mask, X, Y

# ===============================
# 🌈 Blob Rendering
# ===============================
def render_blob(ax, px, py, base_color,
                depth=0.5, light_dir=(-0.6, 0.8),
                shadow=True, blur_strength=0.015):
    from matplotlib.path import Path

    xmin, xmax = px.min(), px.max()
    ymin, ymax = py.min(), py.max()
    pad = 0.4 * max(xmax - xmin, ymax - ymin)
    xmin -= pad; xmax += pad
    ymin -= pad; ymax += pad

    verts = np.column_stack([px, py])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    path = Path(verts, codes)

    H, W = 400, 400
    mask, X, Y = rasterize_path_mask(path, xmin, xmax, ymin, ymax, W, H)
    mask_f = mask.astype(float)

    cx, cy = px.mean(), py.mean()
    nx, ny = X - cx, Y - cy
    dist = np.sqrt(nx ** 2 + ny ** 2)
    radial = np.clip(1.0 - dist / dist.max(), 0, 1)

    light = np.dot(np.stack([nx, ny], axis=-1), np.array(light_dir))
    light = (light - light.min()) / (light.max() - light.min())

    shade = 0.3 + 0.7 * (0.5 * radial + 0.5 * light)
    base = np.array(base_color).reshape((1, 1, 3))
    img_rgb = np.clip(base * shade[..., None], 0, 1)
    alpha = mask_f * (0.4 + 0.6 * depth)

    rgba = np.zeros((H, W, 4))
    rgba[..., :3] = img_rgb
    rgba[..., 3] = alpha

    if shadow:
        shadow_sigma = max(2.0, max(W, H) * blur_strength)
        shadow_mask = separable_gaussian_blur(mask_f, shadow_sigma)
        ax.imshow(shadow_mask, extent=[xmin + 0.02, xmax + 0.02, ymin - 0.02, ymax - 0.02],
                  cmap="gray", alpha=0.25 * (1 - depth), zorder=0)

    ax.imshow(rgba, extent=[xmin, xmax, ymin, ymax], origin="lower", interpolation="bilinear", zorder=1)


# ===============================
# 🖥️ Streamlit Interface
# ===============================
st.set_page_config(page_title="3D Generative Poster", layout="wide")

st.title("🎨 Interactive Generative 3D-like Poster")
st.markdown("Created by **He Pengwei** · Arts & Advanced Big Data · Week 4")

# Sidebar control
st.sidebar.header("Control Panel")

seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=42)
n_blobs = st.sidebar.slider("Number of Blobs", 5, 30, 14)
light_x = st.sidebar.slider("Light X Direction", -1.0, 1.0, -0.6)
light_y = st.sidebar.slider("Light Y Direction", -1.0, 1.0, 0.8)
theme = st.sidebar.selectbox("Background Theme", ["cool", "warm", "neutral"])
shadow_toggle = st.sidebar.checkbox("Enable Shadows", value=True)
blur_strength = st.sidebar.slider("Shadow Blur Strength", 0.005, 0.05, 0.015)

if st.sidebar.button("🎨 Generate Poster"):
    start = time.time()
    random.seed(seed)
    np.random.seed(seed)
    palette = diverse_palette(n_blobs, seed=seed)

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Background color
    bg_colors = {
        "cool": ("#a2cffe", "#f7faff"),
        "warm": ("#ffdfba", "#fffaf0"),
        "neutral": ("#e8e8e8", "#ffffff")
    }
    c1, c2 = bg_colors[theme]
    grad = np.linspace(0, 1, 600).reshape(-1, 1)
    ax.imshow(grad, extent=[0, 1, 0, 1], origin="lower", cmap="coolwarm" if theme == "cool" else "Wistia", alpha=0.2)

    # Generate blobs
    blobs = []
    for i in range(n_blobs):
        r = random.uniform(0.1, 0.26)
        wobble = random.uniform(0.08, 0.22)
        cx, cy = random.uniform(0.12, 0.88), random.uniform(0.12, 0.88)
        angles = np.linspace(0, 2 * np.pi, 280)
        rr = r * (1 + wobble * (np.random.rand(len(angles)) - 0.5) * 2)
        px = cx + rr * np.cos(angles)
        py = cy + rr * np.sin(angles)
        depth = i / float(max(1, n_blobs - 1))
        color = np.array(palette[i % len(palette)])
        render_blob(ax, px, py, color, depth=depth,
                    light_dir=(light_x, light_y),
                    shadow=shadow_toggle,
                    blur_strength=blur_strength)

    ax.text(0.5, 0.03, f"Seed: {seed}", fontsize=10, ha="center", color="gray", transform=ax.transAxes)
    st.pyplot(fig)

    # Export download
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    st.download_button("📥 Download Poster as PNG", data=buf.getvalue(),
                       file_name=f"poster_seed_{seed}.png", mime="image/png")

    st.caption(f"⏱️ Rendered in {time.time() - start:.2f} seconds.")
