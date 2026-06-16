#!/usr/bin/env python3
"""Finite-field plots extracted from finite_field_visualization.ipynb.

Run with Sage's Python, for example:

    sage -python finite_field_plots.py subfield --q 25 --output images/subfield_finite_field_25.jpeg
    sage -python finite_field_plots.py coordinate-axes --q 25 --output images/finite_field_coordinate_axes_25.jpeg
    sage -python finite_field_plots.py projective-lines --q 49 --output images/lines_through_origin_finite_field_49.jpeg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sage.all import GF, I, exp, pi


def ensure_parent(path):
    if path:
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def root_xy(exponent, order):
    root = exp(2 * pi * I * exponent / order)
    return float(root.real()), float(root.imag())


def sign(value):
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def format_field_element(element):
    if element == 0:
        return "0"

    coeffs = element.polynomial().coefficients(sparse=False)
    terms = []
    for i, coeff in enumerate(coeffs):
        if coeff == 0:
            continue
        if i == 0:
            terms.append(str(coeff))
        elif i == 1:
            terms.append("x" if coeff == 1 else f"{coeff}x")
        else:
            terms.append(f"x^{i}" if coeff == 1 else f"{coeff}x^{i}")
    return " + ".join(terms) if terms else "0"


def setup_field_circle(q, title):
    F = GF(q, repr="poly")
    alpha = F.multiplicative_generator()
    order = q - 1
    elements = [alpha**k for k in range(order)]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="datalim")
    theta = np.linspace(0, 2 * np.pi, 1000)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1)

    ax.plot(0, 0, "ko", markersize=5)
    ax.text(0.05, 0, "0", fontsize=8, ha="center", va="center", color="black")

    for k in range(order):
        x, y = root_xy(k, order)
        ax.plot(x, y, "o", color="#5b8fd9", markersize=5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.axis("off")

    return fig, ax, F, alpha, elements


def label_field_elements(ax, elements, q):
    order = q - 1
    for k, element in enumerate(elements):
        x, y = root_xy(k, order)
        y_offset = 0.0
        if abs(y) == 1.0:
            y_offset = 0.1
        elif 0.9 < abs(y) < 1.0:
            y_offset = 0.06
        elif 0.8 < abs(y) <= 0.9:
            y_offset = 0.025

        label = f"$\\alpha^{{{k}}} = {format_field_element(element)}$"
        ax.text(
            x * 1.2,
            y * 1.2 + sign(y) * y_offset,
            label,
            fontsize=8,
            ha="center",
            va="center",
        )


def finish_plot(fig, ax, output_filename=None, show=True, dpi=300):
    ax.legend(loc="upper right")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    if output_filename:
        ensure_parent(output_filename)
        fig.savefig(output_filename, format=Path(output_filename).suffix.lstrip(".") or "png", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_subfield_finite_field(q, output_filename=None, show=True, labels=True, dpi=300):
    F = GF(q, repr="poly")
    p = F.characteristic()

    fig, ax, _, _, elements = setup_field_circle(q, f"Subfield $F_{{{p}}}$ of $F_{{{q}}}$")
    if labels:
        label_field_elements(ax, elements, q)

    step = (q - 1) // (p - 1)
    subfield_points = [root_xy(k * step, q - 1) for k in range(p - 1)]
    for x, y in subfield_points:
        ax.plot(x, y, "ro", markersize=6)

    x_vals, y_vals = zip(*(subfield_points + [subfield_points[0]]))
    ax.plot(x_vals, y_vals, "r-", linewidth=2, label=f"$F_{{{p}}}$ subfield")

    finish_plot(fig, ax, output_filename, show, dpi)


def plot_coordinate_axes_finite_field(q, output_filename=None, show=True, labels=True, dpi=300):
    F = GF(q, repr="poly")
    p = F.characteristic()
    r = F.degree()
    beta = F.gen()
    prime_field = GF(p)

    fig, ax, _, alpha, elements = setup_field_circle(q, f"Finite Field Coordinate Axes $F_{{{q}}}$")
    if labels:
        label_field_elements(ax, elements, q)

    colors = plt.cm.jet(np.linspace(0, 1, r))
    for i, color in enumerate(colors):
        axis_elements = [F(c) * beta**i for c in prime_field if c != 0]
        exponents = [int(element.log(alpha)) for element in axis_elements]
        points = [root_xy(exponent, q - 1) for exponent in exponents]
        x_vals, y_vals = zip(*(points + [points[0]]))
        ax.plot(x_vals, y_vals, "o-", color=color, linewidth=2, label=f"$x^{i}$ axis")

    finish_plot(fig, ax, output_filename, show, dpi)


def plot_projective_line_finite_field(q, output_filename=None, show=True, labels=True, dpi=300):
    F = GF(q, repr="poly")
    p = F.characteristic()

    fig, ax, _, _, elements = setup_field_circle(q, f"Lines through origin in $F_{{{q}}}$")
    if labels:
        label_field_elements(ax, elements, q)

    num_lines = (q - 1) // (p - 1)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_lines))
    for i, color in enumerate(colors):
        exponents = [i + k * num_lines for k in range(p - 1)]
        points = [root_xy(exponent, q - 1) for exponent in exponents]
        x_vals, y_vals = zip(*(points + [points[0]]))
        ax.plot(x_vals, y_vals, "o-", color=color, linewidth=2, label=f"$F_{{{p}}} \\alpha^{i}$")

    finish_plot(fig, ax, output_filename, show, dpi)


def build_parser():
    parser = argparse.ArgumentParser(description="Finite-field visualization scripts for Sage Python.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_plot_options(subparser, default_q=25):
        subparser.add_argument("--q", type=int, default=default_q, help="Finite field order.")
        subparser.add_argument("--output", help="Output image path.")
        subparser.add_argument("--no-show", action="store_true", help="Save or compute without opening a window.")
        subparser.add_argument("--no-labels", action="store_true", help="Suppress alpha/vector labels.")
        subparser.add_argument("--dpi", type=int, default=300, help="Saved-image DPI.")

    add_plot_options(subparsers.add_parser("subfield", help="Plot the prime subfield inside F_q."))
    add_plot_options(subparsers.add_parser("coordinate-axes", help="Plot prime-field coordinate axes in F_q."))
    add_plot_options(subparsers.add_parser("projective-lines", help="Plot all F_p-lines through zero in F_q."))

    all_parser = subparsers.add_parser("all", help="Generate all finite-field plots for one q.")
    all_parser.add_argument("--q", type=int, default=25, help="Finite field order.")
    all_parser.add_argument("--output-dir", default="images", help="Directory for generated images.")
    all_parser.add_argument("--no-show", action="store_true", help="Save or compute without opening a window.")
    all_parser.add_argument("--no-labels", action="store_true", help="Suppress alpha/vector labels.")
    all_parser.add_argument("--dpi", type=int, default=300, help="Saved-image DPI.")

    return parser


def main():
    args = build_parser().parse_args()
    show = not args.no_show
    labels = not args.no_labels

    if args.command == "subfield":
        plot_subfield_finite_field(args.q, args.output, show, labels, args.dpi)
    elif args.command == "coordinate-axes":
        plot_coordinate_axes_finite_field(args.q, args.output, show, labels, args.dpi)
    elif args.command == "projective-lines":
        plot_projective_line_finite_field(args.q, args.output, show, labels, args.dpi)
    elif args.command == "all":
        output_dir = Path(args.output_dir)
        plot_subfield_finite_field(args.q, output_dir / f"subfield_finite_field_{args.q}.jpeg", show, labels, args.dpi)
        plot_coordinate_axes_finite_field(args.q, output_dir / f"finite_field_coordinate_axes_{args.q}.jpeg", show, labels, args.dpi)
        plot_projective_line_finite_field(args.q, output_dir / f"lines_through_origin_finite_field_{args.q}.jpeg", show, labels, args.dpi)


if __name__ == "__main__":
    main()
