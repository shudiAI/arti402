"""
arti402_figures.py
==================
Diagram helpers for ARTI 402 — Deep Learning (Lab 3 onward).

Keep this file in the SAME FOLDER as the notebook, then import what you need:

    from arti402_figures import (draw_cnn_architecture, draw_convolution_step,
                                 draw_max_pooling, draw_lab_pipeline,
                                 draw_param_example)

Every figure is drawn from the same numbers the notebook computes, so the
pictures can never disagree with the code.

The architecture diagram will use a real photograph as its input block if one
is found next to this file (default: "Zebra_image.jpeg"). If no image is
found it falls back to a plain grey rectangle, so the notebook always runs.

You do not need to read or understand this file to complete the lab.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

__all__ = [
    "draw_cnn_architecture", "draw_convolution_step", "draw_max_pooling",
    "draw_shape_journey", "draw_lab_pipeline", "draw_param_example",
    "INPUT_IMAGE_CANDIDATES",
    # Lab 4
    "draw_unrolled_rnn", "draw_repeated_multiplication", "draw_lstm_cell",
    "draw_recurrent_params", "draw_lab4_pipeline",
]

# ----------------------------------------------------------------- palette
LEARN = "#E8A33D"     # blocks WITH learnable parameters
FIXED = "#8FB8D6"     # blocks with NO parameters
EDGE = "#333333"
GREY = "#DCDCDC"
OUTC = "#F3D9B1"

# Filenames tried, in order, for the architecture diagram's input block.
INPUT_IMAGE_CANDIDATES = [
    "Zebra_image.jpeg", "Zebra_image.jpg", "Zebra_image.png", "Zebra_image",
]


# ----------------------------------------------------------------- helpers
def _find_input_image(path=None):
    """Return a path to the input photo, or None if there isn't one."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [path] if path else []
    candidates += INPUT_IMAGE_CANDIDATES
    for name in candidates:
        if not name:
            continue
        for full in (name, os.path.join(here, name)):
            if os.path.isfile(full):
                return full
    return None


def _center_crop(img, target_aspect):
    """Centre-crop an (H, W, C) array to the given width/height aspect ratio."""
    h, w = img.shape[:2]
    if w / h > target_aspect:                 # too wide -> trim the sides
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    new_h = int(round(w / target_aspect))     # too tall -> trim top/bottom
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, :]


def _stack(ax, x, y, w, h, n, color, dx=.055, dy=.045):
    """n offset rectangles, to suggest a stack of feature maps."""
    for i in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + i * dx, y + i * dy), w, h, facecolor=color,
                               edgecolor=EDGE, lw=1.1, zorder=10 - i))


def _arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#555"))


# ------------------------------------------------------------------ FIG 1
def draw_cnn_architecture(image_path=None, verbose=True):
    """The four building blocks, end to end.

    image_path : optional path to a photo for the input block.
    """
    fig, ax = plt.subplots(figsize=(13.5, 4.9))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(-1.5, 3.9)
    ax.axis("off")
    mid, BLK_Y = 1.5, 2.92

    # ---- input block: a real photo if we can find one -------------------
    ix, iw, ih = .15, 1.35, 1.55
    iy0, iy1 = mid - ih / 2, mid + ih / 2
    found = _find_input_image(image_path)
    if found is not None:
        photo = plt.imread(found)
        if photo.ndim == 2:
            photo = np.stack([photo] * 3, axis=-1)
        photo = _center_crop(photo[:, :, :3], iw / ih)
        ax.imshow(photo, extent=(ix, ix + iw, iy0, iy1),
                  aspect="auto", origin="upper", zorder=2)
    else:
        ax.add_patch(Rectangle((ix, iy0), iw, ih, facecolor=GREY,
                               edgecolor=EDGE, lw=1.1, zorder=2))
        ax.text(ix + iw / 2, mid, "image", ha="center", va="center",
                fontsize=9, color="#666", zorder=3)
        if verbose:
            print("note: no input photo found, using a grey placeholder.\n"
                  "      put Zebra_image.jpeg next to the notebook to show a real image.")

    ax.add_patch(Rectangle((ix, iy0), iw, ih, facecolor="none",
                           edgecolor=EDGE, lw=1.3, zorder=4))
    ax.text(ix + iw / 2, iy1 + .16, "Input", ha="center", fontsize=11,
            weight="bold")

    # the little kernel window, placed over the stripes
    ks = .26
    kx, ky = ix + .42, mid + .16
    ax.add_patch(Rectangle((kx, ky), ks, ks, facecolor="none",
                           edgecolor="#FFD24A", lw=2.4, zorder=6))
    ax.text(kx + ks / 2, ky + ks + .10, "kernel", ha="center", va="center",
            fontsize=7.4, color="#FFFFFF", zorder=7,
            bbox=dict(boxstyle="round,pad=0.16", fc="#000000AA", ec="none"))

    _arrow(ax, ix + iw + .10, 1.85, mid)

    # ---- convolution / pooling stacks -----------------------------------
    blocks = [(1.95, .95, 1.75, 5, LEARN, "Convolution\n+ ReLU", "BLOCK 1"),
              (3.75, 1.08, 1.50, 5, FIXED, "Pooling", "BLOCK 2"),
              (5.45, .80, 1.20, 8, LEARN, "Convolution\n+ ReLU", "BLOCK 1"),
              (7.05, .92, .95, 8, FIXED, "Pooling", "BLOCK 2")]
    for i, (x, w, h, n, col, lbl, blk) in enumerate(blocks):
        _stack(ax, x, mid - h / 2, w, h, n, col)
        ax.text(x + w / 2 + .12, mid - h / 2 - .42, lbl, ha="center",
                fontsize=8.6, linespacing=1.25)
        ax.text(x + w / 2 + .12, BLK_Y, blk, ha="center", fontsize=7.2,
                color="#777", style="italic")
        if i < 3:
            _arrow(ax, x + w + .42, blocks[i + 1][0] - .08, mid)
    _arrow(ax, 8.35, 8.85, mid)

    # ---- flatten --------------------------------------------------------
    ax.add_patch(Rectangle((8.95, .35), .26, 2.3, facecolor=FIXED,
                           edgecolor=EDGE, lw=1.1))
    for k in range(1, 10):
        ax.plot([8.95, 9.21], [.35 + k * .23] * 2, color=EDGE, lw=.6)
    ax.text(9.08, .05, "Flatten", ha="center", fontsize=8.6)
    ax.text(9.08, BLK_Y, "BLOCK 3", ha="center", fontsize=7.2, color="#777",
            style="italic")

    # ---- fully connected ------------------------------------------------
    L1 = [(10.15, .45 + i * .29) for i in range(8)]
    L2 = [(11.15, .85 + i * .33) for i in range(5)]
    L3 = [(12.15, 1.05), (12.15, 1.5), (12.15, 1.95)]
    for a in L1:
        for b in L2:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#C9C9C9", lw=.35, zorder=1)
    for b in L2:
        for c in L3:
            ax.plot([b[0], c[0]], [b[1], c[1]], color="#C9C9C9", lw=.35, zorder=1)
    for (px, py) in L1 + L2 + L3:
        ax.add_patch(plt.Circle((px, py), .085, facecolor=LEARN,
                                edgecolor=EDGE, lw=.7, zorder=5))
    ax.text(11.15, .05, "Fully connected", ha="center", fontsize=8.6)
    ax.text(11.15, BLK_Y, "BLOCK 4", ha="center", fontsize=7.2, color="#777",
            style="italic")
    _arrow(ax, 12.35, 12.85, mid)

    # ---- softmax output -------------------------------------------------
    for i, (nm, p) in enumerate([("Horse", .2), ("Zebra", .7), ("Dog", .1)]):
        yy = 2.12 - i * .55
        ax.add_patch(Rectangle((12.95, yy - .2), .85, .42, facecolor=OUTC,
                               edgecolor=EDGE, lw=1))
        ax.text(13.37, yy - .02, f"{p}", ha="center", va="center", fontsize=9)
        ax.text(13.95, yy - .02, nm, ha="left", va="center", fontsize=9,
                weight="bold" if p == .7 else "normal")
    ax.text(13.37, BLK_Y, "Softmax", ha="center", fontsize=8.2, color="#777",
            style="italic")

    # ---- brackets and legend -------------------------------------------
    def bracket(x0, x1, label):
        y = -.72
        ax.plot([x0, x0, x1, x1], [y + .16, y, y, y + .16], color="#666", lw=1.1)
        ax.text((x0 + x1) / 2, y - .42, label, ha="center", fontsize=9.6,
                weight="bold", color="#333")
    bracket(.15, 8.15, "Feature extraction")
    bracket(8.85, 12.45, "Classification")
    bracket(12.9, 14.9, "Probabilities")

    ax.add_patch(Rectangle((.3, 3.5), .36, .24, facecolor=LEARN,
                           edgecolor=EDGE, lw=1))
    ax.text(.78, 3.62, "has learnable parameters", va="center", fontsize=9)
    ax.add_patch(Rectangle((5.1, 3.5), .36, .24, facecolor=FIXED,
                           edgecolor=EDGE, lw=1))
    ax.text(5.58, 3.62, "no parameters (fixed operation)", va="center", fontsize=9)

    ax.set_title("The four building blocks of a CNN", fontsize=14,
                 weight="bold", pad=2)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ FIG 2
def draw_convolution_step():
    """One step of convolution, with the arithmetic spelled out."""
    img = np.array([[1, 2, 3, 0, 1], [4, 5, 6, 1, 2], [7, 8, 9, 2, 0],
                    [1, 1, 1, 1, 3], [2, 0, 1, 4, 1]], float)
    ker = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)

    fig, ax = plt.subplots(figsize=(12.2, 4.1))
    ax.set_xlim(0, 17.5)
    ax.set_ylim(-.9, 6.5)
    ax.axis("off")
    ax.invert_yaxis()

    def grid(x0, y0, M, cell=.78, hl=None, fs=9, cmap=None):
        r, c = M.shape
        for i in range(r):
            for j in range(c):
                inside = hl is not None and hl[0] <= i < hl[0] + 3 and hl[1] <= j < hl[1] + 3
                fc = "#FDE8C3" if inside else ("#FFFFFF" if cmap is None else cmap(i, j))
                ax.add_patch(Rectangle((x0 + j * cell, y0 + i * cell), cell, cell,
                                       facecolor=fc, edgecolor="#999", lw=.8))
                ax.text(x0 + j * cell + cell / 2, y0 + i * cell + cell / 2,
                        f"{M[i, j]:.0f}", ha="center", va="center", fontsize=fs)
        if hl is not None:
            ax.add_patch(Rectangle((x0 + hl[1] * cell, y0 + hl[0] * cell),
                                   3 * cell, 3 * cell, facecolor="none",
                                   edgecolor="#D4820A", lw=2.4, zorder=6))

    grid(.3, .6, img, hl=(1, 1))
    ax.text(2.3, -.15, "input  5 x 5", fontsize=10, weight="bold")
    ax.text(6.0, 2.6, "x", fontsize=16, ha="center", va="center")
    grid(6.8, 1.4, ker)
    ax.text(7.97, .65, "kernel  3 x 3", fontsize=10, weight="bold", ha="center")
    ax.text(10.1, 2.6, "=", fontsize=16, ha="center", va="center")

    out = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            out[i, j] = np.sum(img[i:i + 3, j:j + 3] * ker)
    grid(11.0, 1.4, out, fs=8.5,
         cmap=lambda i, j: "#FDE8C3" if (i, j) == (1, 1) else "#FFFFFF")
    ax.add_patch(Rectangle((11.78, 2.18), .78, .78, facecolor="none",
                           edgecolor="#D4820A", lw=2.4, zorder=6))
    ax.text(12.17, .65, "output  3 x 3", fontsize=10, weight="bold", ha="center")

    patch = img[1:4, 1:4]
    terms = " + ".join(f"({int(p)})({int(k)})"
                       for p, k in zip(patch.ravel(), ker.ravel()))
    ax.text(8.7, 5.55, f"the highlighted output  =  {terms}  =  {out[1, 1]:.0f}",
            fontsize=9.2, ha="center", color="#333")
    ax.text(8.7, 6.05, "slide the window one step and repeat", fontsize=9.5,
            ha="center", style="italic", color="#666")
    ax.set_title("Convolution: multiply the patch by the kernel, sum, slide",
                 fontsize=12.5, weight="bold", pad=4)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ FIG 3
def draw_max_pooling():
    """2x2 max pooling, with each tile colour-matched to its output cell."""
    M = np.arange(1, 17, dtype=float).reshape(4, 4)
    cols = ["#F6C6C6", "#C6DFF6", "#CFEFCF", "#F2E1B8"]

    fig, ax = plt.subplots(figsize=(10.4, 3.5))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(-1.2, 4.6)
    ax.axis("off")
    ax.invert_yaxis()
    cell = .86

    for i in range(4):
        for j in range(4):
            t = (i // 2) * 2 + (j // 2)
            ax.add_patch(Rectangle((.4 + j * cell, .5 + i * cell), cell, cell,
                                   facecolor=cols[t], edgecolor="#888", lw=.8))
            ismax = (i % 2 == 1 and j % 2 == 1)
            ax.text(.4 + j * cell + cell / 2, .5 + i * cell + cell / 2,
                    f"{M[i, j]:.0f}", ha="center", va="center", fontsize=10.5,
                    weight="bold" if ismax else "normal",
                    color="#000" if ismax else "#666")
    for t in range(4):
        i, j = (t // 2) * 2, (t % 2) * 2
        ax.add_patch(Rectangle((.4 + j * cell, .5 + i * cell), 2 * cell, 2 * cell,
                               facecolor="none", edgecolor="#444", lw=2.2))

    ax.text(.4 + 2 * cell, .12, "feature map  4 x 4", fontsize=10,
            weight="bold", ha="center")
    ax.text(.4 + 2 * cell, 4.35, "tiles do NOT overlap", fontsize=8.8,
            ha="center", style="italic", color="#666")

    ax.annotate("", xy=(7.6, 2.2), xytext=(4.35, 2.2),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#555"))
    ax.text(5.95, 1.82, "keep the max\nof each tile", fontsize=9.4,
            ha="center", linespacing=1.3)

    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((8.1 + j * cell, 1.35 + i * cell), cell, cell,
                                   facecolor=cols[i * 2 + j], edgecolor="#444", lw=2))
            ax.text(8.1 + j * cell + cell / 2, 1.35 + i * cell + cell / 2,
                    f"{M[i * 2 + 1, j * 2 + 1]:.0f}", ha="center", va="center",
                    fontsize=11.5, weight="bold")
    ax.text(8.1 + cell, .97, "output  2 x 2", fontsize=10, weight="bold",
            ha="center")
    ax.text(11.1, 2.2, "75% of the numbers\nare gone, and the\nstrongest signal\nis kept",
            fontsize=9.2, va="center", linespacing=1.5, color="#333")
    ax.set_title("Max pooling", fontsize=12.5, weight="bold", pad=4)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------- FIG 4/5
def draw_shape_journey(stages, title, figw=13.0):
    """A horizontal chain of blocks showing shape and parameter count.

    stages : list of (label, shape_text, params, kind)
             params : int, str, or None ;  kind : input | learn | fixed | out
    """
    n = len(stages)
    fig, ax = plt.subplots(figsize=(figw, 3.5))
    ax.set_xlim(0, n * 2.25)
    ax.set_ylim(-1.35, 1.75)
    ax.axis("off")
    colmap = {"input": GREY, "learn": LEARN, "fixed": FIXED, "out": OUTC}

    for i, (lbl, shp, par, kind) in enumerate(stages):
        x, w = i * 2.25 + .16, 1.72
        ax.add_patch(Rectangle((x, -.30), w, .78, facecolor=colmap[kind],
                               edgecolor=EDGE, lw=1.2))
        ax.text(x + w / 2, .09, lbl, ha="center", va="center", fontsize=9.3,
                weight="bold", linespacing=1.25)
        ax.text(x + w / 2, -.56, shp, ha="center", va="top", fontsize=9.6,
                family="monospace", color="#1a1a1a")
        if par is None:
            ax.text(x + w / 2, .76, "—", ha="center", fontsize=9, color="#888")
        elif isinstance(par, str):
            ax.text(x + w / 2, .76, par, ha="center", fontsize=8.8,
                    color="#9A6000", weight="bold")
        else:
            ax.text(x + w / 2, .76, f"{par:,} params" if par else "0 params",
                    ha="center", fontsize=8.8,
                    color="#9A6000" if par else "#5A7A93",
                    weight="bold" if par else "normal")
        if i < n - 1:
            ax.annotate("", xy=(x + w + .38, .09), xytext=(x + w + .03, .09),
                        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#555"))

    ax.text(n * 2.25 / 2, 1.45, title, ha="center", fontsize=12.5, weight="bold")
    ax.text(.16, -1.12, "shape", fontsize=8.5, color="#888", style="italic")
    ax.text(.16, 1.02, "learnable parameters", fontsize=8.5, color="#888",
            style="italic")
    plt.tight_layout()
    return fig


def draw_lab_pipeline():
    """The exact pipeline built in this lab."""
    return draw_shape_journey([
        ("input\nimage", "(12, 12)", None, "input"),
        ("BLOCK 1\nconv 3x3 x4 + ReLU", "(10, 10, 4)", "40 (frozen)", "learn"),
        ("BLOCK 2\nmax pool, tile 10", "(1, 1, 4)", 0, "fixed"),
        ("BLOCK 3\nflatten", "(4,)", 0, "fixed"),
        ("BLOCK 4\ndense -> softmax", "(3,)", 15, "out"),
    ], "This lab's pipeline: only BLOCK 4 is trained", figw=12.0)


def draw_param_example():
    """The two-conv-layer network students count parameters for."""
    return draw_shape_journey([
        ("input\nRGB image", "32x32x3", None, "input"),
        ("conv 3x3 x10\n+ ReLU", "30x30x10", 280, "learn"),
        ("max pool\ntile 2", "15x15x10", 0, "fixed"),
        ("conv 3x3 x20\n+ ReLU", "13x13x20", 1820, "learn"),
        ("max pool\ntile 2", "6x6x20", 0, "fixed"),
        ("flatten", "(720,)", 0, "fixed"),
        ("dense\n-> 3 classes", "(3,)", 2163, "out"),
    ], "Exercise 5: trace the shapes, count the parameters  (total 4,263)",
       figw=15.0)


# =====================================================================
#                              LAB 4 FIGURES
# =====================================================================
STAGE1 = "#E57373"   # forget      (StatQuest stage 1)
STAGE2 = "#81C784"   # input       (StatQuest stage 2)
STAGE3 = "#64B5F6"   # output      (StatQuest stage 3)
CELLC = "#FFE0B2"    # recurrent cell


def _box(ax, x, y, w, h, fc, text="", fs=10, bold=True, ec=EDGE, lw=1.3, z=5):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                weight="bold" if bold else "normal", zorder=z + 1)


def _arr(ax, x0, y0, x1, y1, color="#555", lw=1.4, z=3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0), zorder=z)


def _node(ax, x, y, sym, fc="#FFFFFF", r=.22, fs=12):
    ax.add_patch(plt.Circle((x, y), r, facecolor=fc, edgecolor=EDGE, lw=1.3, zorder=8))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs, weight="bold", zorder=9)


# ------------------------------------------------------------------ FIG 6
def draw_unrolled_rnn(n_steps=4):
    """The recurrent loop, and the same cell unrolled through time."""
    fig, ax = plt.subplots(figsize=(13.4, 4.2))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(-.3, 4.4)
    ax.axis("off")

    # rolled version
    _box(ax, .6, 1.5, 1.5, 1.1, CELLC, "RNN\ncell", fs=10)
    _arr(ax, 1.35, .55, 1.35, 1.5)
    ax.text(1.35, .3, "x", ha="center", fontsize=12, style="italic")
    _arr(ax, 1.35, 2.6, 1.35, 3.45)
    ax.text(1.35, 3.62, "h", ha="center", fontsize=12, style="italic")
    ax.annotate("", xy=(2.1, 1.85), xytext=(2.1, 2.3),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#C0392B",
                                connectionstyle="arc3,rad=-2.6"), zorder=4)
    ax.text(3.05, 2.05, "loop:\nh feeds\nback in", fontsize=8.6, color="#C0392B",
            va="center", linespacing=1.2)
    ax.text(4.15, 2.05, "=", fontsize=22, ha="center", va="center")

    # unrolled version
    x0, gap, w = 6.25, 2.4, 1.45
    ax.text(x0 - .55, 2.05, "h₀ = 0", ha="right", va="center", fontsize=9.5,
            color="#555")
    _arr(ax, x0 - .5, 2.05, x0, 2.05, color="#C0392B")
    for t in range(n_steps):
        x = x0 + t * gap
        _box(ax, x, 1.5, w, 1.1, CELLC, "RNN\ncell", fs=10)
        _arr(ax, x + w / 2, .55, x + w / 2, 1.5)
        ax.text(x + w / 2, .3, f"x{t + 1}", ha="center", fontsize=11, style="italic")
        _arr(ax, x + w / 2, 2.6, x + w / 2, 3.45)
        ax.text(x + w / 2, 3.62, f"h{t + 1}", ha="center", fontsize=11, style="italic")
        if t < n_steps - 1:
            _arr(ax, x + w, 2.05, x + gap, 2.05, color="#C0392B")
            ax.text(x + w + (gap - w) / 2, 2.2, f"h{t + 1}", ha="center",
                    fontsize=8.5, color="#C0392B")
    ax.text(x0 + (n_steps - 1) * gap + w + .25, 2.05, "...", fontsize=16,
            va="center")

    ax.text(x0 + ((n_steps - 1) * gap + w) / 2, -.12,
            "every copy is the SAME cell with the SAME weights Wx, Wh, b",
            ha="center", fontsize=10, weight="bold", color="#8A4B00")
    ax.set_title("An RNN is one cell, applied once per time step",
                 fontsize=13, weight="bold", pad=4)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ FIG 7
def draw_repeated_multiplication(steps=20):
    """StatQuest's point: multiplying by the same weight again and again."""
    n = np.arange(steps + 1)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for w, col, note in [(1.5, "#C0392B", "explodes"), (1.0, "#555555", "stays"),
                         (0.9, "#E67E22", "fades"), (0.5, "#2471A3", "vanishes")]:
        ax.plot(n, w ** n, lw=2.4, color=col, label=f"w = {w}   ({note})")
    ax.set_yscale("log")
    ax.set_xlabel("number of time steps back")
    ax.set_ylabel("w ^ steps   (log scale)")
    ax.set_title("Multiply by the same weight at every step", fontsize=12.5,
                 weight="bold")
    ax.axhline(1, color="grey", lw=.8, ls="--")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=9, loc="center left")
    ax.text(20, 0.5 ** 20 * 3, f"0.5^20 = {0.5 ** 20:.1e}", ha="right",
            fontsize=8.5, color="#2471A3")
    ax.text(20, 1.5 ** 20 / 3, f"1.5^20 = {1.5 ** 20:,.0f}", ha="right",
            va="top", fontsize=8.5, color="#C0392B")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ FIG 8
def draw_lstm_cell():
    """One LSTM cell, gates coloured by StatQuest's three stages."""
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    ax.set_xlim(-.4, 13.3)
    ax.set_ylim(-1.05, 6.4)
    ax.axis("off")

    # cell outline
    ax.add_patch(Rectangle((1.2, .5), 9.4, 5.0, facecolor="#FAFAFA",
                           edgecolor="#999", lw=1.2, ls="--", zorder=0))

    # stage shading
    for x, w, col, txt in [(1.5, 2.3, STAGE1, "STAGE 1\nhow much to REMEMBER"),
                           (4.0, 3.2, STAGE2, "STAGE 2\nupdate LONG-term memory"),
                           (7.4, 3.0, STAGE3, "STAGE 3\nupdate SHORT-term memory")]:
        ax.add_patch(Rectangle((x, .65), w, 3.55, facecolor=col, alpha=.16,
                               edgecolor="none", zorder=0))
        ax.text(x + w / 2, .88, txt, ha="center", fontsize=8.3, weight="bold",
                color="#333", linespacing=1.2, zorder=1)

    TOP, GATE, BOT = 4.75, 2.55, 1.55

    # long-term memory line (cell state)
    ax.plot([-.1, 11.9], [TOP, TOP], color="#8A4B00", lw=3.2, zorder=2)
    ax.text(-.1, TOP + .32, "c  (t-1)", fontsize=10, color="#8A4B00", weight="bold")
    ax.text(12.0, TOP + .32, "c  (t)", fontsize=10, color="#8A4B00", weight="bold")
    _arr(ax, 11.4, TOP, 12.0, TOP, color="#8A4B00", lw=2.4)
    ax.text(6.0, 5.85, "LONG-TERM MEMORY  (cell state c)  —  the conveyor belt",
            ha="center", fontsize=10.5, weight="bold", color="#8A4B00")

    # short-term memory in (h_{t-1}) and input x_t
    ax.plot([-.1, 8.3], [BOT, BOT], color="#1F4E79", lw=2.2, zorder=2)
    ax.text(-.1, BOT - .42, "h  (t-1)", fontsize=10, color="#1F4E79", weight="bold")
    _arr(ax, 1.9, -.55, 1.9, BOT, color="#1F4E79")
    ax.text(1.9, -.78, "x  (t)", ha="center", fontsize=10, weight="bold")
    ax.add_patch(plt.Circle((1.9, BOT), .07, color="#1F4E79", zorder=6))

    # the four gates
    gates = [(2.3, "f", "σ", STAGE1), (4.35, "i", "σ", STAGE2),
             (5.85, "g", "tanh", STAGE2), (7.75, "o", "σ", STAGE3)]
    for gx, name, act, col in gates:
        _arr(ax, gx + .4, BOT, gx + .4, GATE - .02, color="#1F4E79", lw=1.1)
        _box(ax, gx, GATE, .8, .62, col, f"{act}", fs=10 if act == "σ" else 8.6)
        ax.text(gx + .4, GATE - .28, name, ha="center", fontsize=10,
                style="italic", weight="bold", zorder=7)

    # stage 1: forget  ->  x on the belt
    _node(ax, 2.7, TOP, "×")
    _arr(ax, 2.7, GATE + .62, 2.7, TOP - .22, color="#B03A2E", lw=1.6)

    # stage 2: i * g  ->  +  on the belt
    _node(ax, 5.35, 3.65, "×", r=.2, fs=11)
    _arr(ax, 4.75, GATE + .62, 5.2, 3.5, color="#2E7D32", lw=1.5)
    _arr(ax, 6.25, GATE + .62, 5.5, 3.5, color="#2E7D32", lw=1.5)
    _node(ax, 5.35, TOP, "+")
    _arr(ax, 5.35, 3.85, 5.35, TOP - .22, color="#2E7D32", lw=1.6)

    # stage 3: tanh(c) * o  ->  h_t
    ax.plot([9.3, 9.3], [TOP, 3.95], color="#8A4B00", lw=1.6, zorder=2)
    _box(ax, 8.95, 3.35, .7, .55, "#FFFFFF", "tanh", fs=8.4)
    _node(ax, 9.3, 2.86, "×", r=.2, fs=11)
    ax.plot([9.3, 9.3], [3.35, 3.06], color="#555", lw=1.4, zorder=3)
    _arr(ax, 8.55, GATE + .31, 9.1, 2.86, color="#1565C0", lw=1.6)
    ax.plot([9.3, 9.3], [2.66, BOT], color="#1F4E79", lw=2.2, zorder=2)
    ax.plot([9.3, 11.5], [BOT, BOT], color="#1F4E79", lw=2.2, zorder=2)
    _arr(ax, 11.4, BOT, 12.0, BOT, color="#1F4E79", lw=2.2)
    ax.text(12.0, BOT - .42, "h  (t)", fontsize=10, color="#1F4E79", weight="bold")
    ax.text(12.05, BOT + .2, "SHORT-TERM\nMEMORY (h)", fontsize=8.3,
            color="#1F4E79", weight="bold", linespacing=1.15)
    # the two update equations, as a caption
    ax.text(7.2, -.62, "c(t) = f · c(t-1)  +  i · g          h(t) = o · tanh( c(t) )",
            ha="center", fontsize=10.5, family="monospace", color="#222",
            bbox=dict(boxstyle="round,pad=0.45", fc="#FFFFFF", ec="#BBB"))
    ax.set_title("Inside an LSTM cell", fontsize=13.5, weight="bold", pad=2)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ FIG 9
def draw_recurrent_params(n_in, n_hidden):
    """Parameter count of RNN vs GRU vs LSTM for the same sizes."""
    one = n_in * n_hidden + n_hidden * n_hidden + n_hidden
    names = ["RNN", "GRU", "LSTM"]
    mult = [1, 3, 4]
    vals = [m * one for m in mult]
    cols = [CELLC, STAGE2, STAGE3]
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    bars = ax.bar(names, vals, color=cols, edgecolor=EDGE, lw=1.2, width=.55)
    for b, v, m in zip(bars, vals, mult):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * .02,
                f"{v:,}\n({m} × {one})", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_ylabel("learnable parameters")
    ax.set_title(f"Same sizes (inputs {n_in}, hidden {n_hidden}): "
                 f"1, 3 and 4 copies of one block", fontsize=11.5, weight="bold")
    ax.grid(axis="y", alpha=.3)
    ax.text(.02, .96, "RNN: 1 block    GRU: 2 gates + candidate = 3    "
                      "LSTM: 3 gates + candidate = 4",
            transform=ax.transAxes, fontsize=8.4, va="top", color="#555")
    plt.tight_layout()
    return fig


def draw_lab4_pipeline():
    """The exact pipeline used in Lab 4's assessment."""
    return draw_shape_journey([
        ("input\nsequence", "(T, 4)", None, "input"),
        ("recurrent cell\nRNN or LSTM", "h: (12,)", "frozen", "learn"),
        ("keep only the\nLAST h", "(12,)", 0, "fixed"),
        ("dense\n-> softmax", "(3,)", 39, "out"),
    ], "Lab 4 pipeline: read the final hidden state, train only the head",
       figw=11.0)
