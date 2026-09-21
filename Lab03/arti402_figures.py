"""
arti402_figures.py
==================
Diagram helpers for ARTI 402 — Deep Learning, Lab 3 (CNN architecture).

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
