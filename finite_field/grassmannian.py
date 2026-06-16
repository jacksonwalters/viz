#!/usr/bin/env python3
"""Grassmannian utilities extracted from finite_field_visualization.ipynb.

Run with Sage's Python, for example:

    sage -python grassmannian.py count --q 2 --r 4
    sage -python grassmannian.py schubert --q 5 --r 4 --sizes
    sage -python grassmannian.py graph --q 2 --r 4 --k 2 --output images/grassmann_graph_2_4_2.png
    sage -python grassmannian.py subspace --q 5 --r 4 --index 0 --output images/gr_2_4_subspace_0.png
"""

import argparse
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sage.all import GF, I, VectorSpace, exp, graphs, matrix, pi, vector
from sage.matrix.echelon_matrix import reduced_echelon_matrix_iterator


def ensure_parent(path):
    if path:
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def dlog(element, field):
    return int(element.log(field.multiplicative_generator())) if element != 0 else -1


def root_xy(exponent, order):
    root = exp(2 * pi * I * exponent / order)
    return float(root.real()), float(root.imag())


def grassmannian_rref(q, r, k=2):
    """Return RREF matrix representatives for Gr(k, r)(F_q)."""
    F = GF(q)
    return list(reduced_echelon_matrix_iterator(F, k, r, copy=True, set_immutable=True))


def gr_2_naive(q, r):
    """Notebook version: compute Gr(2, r)(F_q) via GF(q^r) element pairs."""
    Fq = GF(q)
    extension = GF(q**r)
    unique_bases = set()

    for a in extension:
        for b in extension:
            if a == 0 or b == 0:
                continue
            if any(a == scalar * b for scalar in Fq):
                continue
            basis = matrix(Fq, [a.list(), b.list()]).echelon_form()
            basis.set_immutable()
            unique_bases.add(basis)

    return list(unique_bases)


def schubert_cells_2(q, r):
    all_cells = {}
    F = GF(q)
    indices = list(combinations(range(r), 2))

    for p0, p1 in indices:
        all_cells[(p0, p1)] = []
        base = matrix(F, 2, r)
        base[:, p0] = vector(F, [1, 0])
        base[:, p1] = vector(F, [0, 1])

        free_param_positions = [j for j in range(r) if j not in (p0, p1)]

        def num_free_params(j):
            if p0 < j < p1:
                return 1
            if j > p1:
                return 2
            return 0

        free_param_counts = [num_free_params(j) for j in free_param_positions]
        for params in product(*[list(F)] * sum(free_param_counts)):
            filled = matrix(base)
            param_iter = iter(params)

            for j in free_param_positions:
                if p0 < j < p1:
                    filled[0, j] = next(param_iter)
                if j > p1:
                    filled[:, j] = vector(F, [next(param_iter), next(param_iter)])

            all_cells[(p0, p1)].append(filled)

    return all_cells


def subspace_finite_field(q, r, basis):
    """Convert the F_q-row span of a basis matrix into elements of GF(q^r).

    This matches the notebook's finite-field-circle model and is intended for
    prime q, where GF(q^r) has an r-term vector representation over GF(q).
    """
    if GF(q).degree() != 1:
        raise ValueError("subspace_finite_field currently expects q to be prime.")

    extension = GF(q**r)
    V = VectorSpace(GF(q), r)
    span = V.subspace(basis)
    return [extension(v.list()) for v in span]


def projective_lines_in_plane(q, r, basis):
    """Return the q + 1 one-dimensional subspaces inside a 2-plane."""
    if basis.nrows() != 2:
        raise ValueError("basis must be a 2-row matrix.")

    F = GF(q)
    extension = GF(q**r)
    row0, row1 = basis.rows()
    directions = [(F(1), t) for t in F] + [(F(0), F(1))]
    lines = []

    for a, b in directions:
        direction = a * row0 + b * row1
        line = [extension((c * direction).list()) for c in F if c != 0]
        lines.append(line)

    return lines


def plot_2d_subspace_finite_field(q, r=4, index=0, output_filename=None, show=True, labels=False, dpi=300):
    """Plot one point of Gr(2, r)(F_q) as q+1 colored projective lines."""
    if GF(q).degree() != 1:
        raise ValueError("plot_2d_subspace_finite_field currently expects q to be prime.")

    representatives = grassmannian_rref(q, r, k=2)
    basis = representatives[index % len(representatives)]
    extension = GF(q**r, repr="poly")
    alpha = extension.multiplicative_generator()
    order = q**r - 1

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="datalim")
    theta = np.linspace(0, 2 * np.pi, 1000)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1)

    for exponent in range(order):
        x, y = root_xy(exponent, order)
        ax.plot(x, y, "o", color="#d6d6d6", markersize=4)

    ax.plot(0, 0, "ko", markersize=5)
    ax.text(0.05, 0, "0", fontsize=8, ha="center", va="center", color="black")

    colors = plt.cm.rainbow(np.linspace(0, 1, q + 1))
    for line_index, (line, color) in enumerate(zip(projective_lines_in_plane(q, r, basis), colors)):
        points = [root_xy(dlog(element, extension), order) for element in line]
        x_vals, y_vals = zip(*(points + [points[0]]))
        ax.plot(x_vals, y_vals, "o-", color=color, linewidth=2, markersize=7, label=f"line {line_index}")

        if labels:
            for element, (x, y) in zip(line, points):
                ax.text(x * 1.12, y * 1.12, f"$\\alpha^{{{dlog(element, extension)}}}$", fontsize=8, ha="center")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"Element {index % len(representatives)} of $Gr(2,{r})(F_{{{q}}})$", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.axis("off")
    ax.legend(loc="upper right")

    if output_filename:
        ensure_parent(output_filename)
        fig.savefig(output_filename, format=Path(output_filename).suffix.lstrip(".") or "png", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_grassmann_graph(q, r=4, k=2, output_filename=None, show=True):
    graph = graphs.GrassmannGraph(q, r, k)
    graph_plot = graph.plot(vertex_labels=False)

    if output_filename:
        ensure_parent(output_filename)
        graph_plot.save(output_filename)

    if show:
        graph_plot.show()

    return graph


def projective_geometry_blocks(q, n=3, d=1, point_coordinates=True):
    """Notebook construction of the projective geometry incidence design."""
    F = GF(q)
    points = {
        p: i
        for i, p in enumerate(
            reduced_echelon_matrix_iterator(F, 1, n + 1, copy=True, set_immutable=True)
        )
    }
    blocks = []

    for m1 in reduced_echelon_matrix_iterator(F, d + 1, n + 1, copy=False):
        block = []
        for m2 in reduced_echelon_matrix_iterator(F, 1, d + 1, copy=False):
            point = m2 * m1
            point.echelonize()
            point.set_immutable()
            block.append(points[point])
        blocks.append(block)

    if not point_coordinates:
        return points, blocks

    inverse_points = {i: p[0] for p, i in points.items()}
    coordinate_blocks = [[inverse_points[i] for i in block] for block in blocks]
    return inverse_points, coordinate_blocks


def print_count(q, r, k):
    representatives = grassmannian_rref(q, r, k)
    print(f"|Gr({k},{r})(F_{q})| = {len(representatives)}")
    if k == 2:
        cells = schubert_cells_2(q, r)
        print("Schubert cell sizes:")
        for pivots, matrices in cells.items():
            print(f"  {pivots}: {len(matrices)}")


def print_schubert(q, r, sizes_only=True):
    cells = schubert_cells_2(q, r)
    for pivots, matrices in cells.items():
        print(f"Pivot columns {pivots}: {len(matrices)} element(s)")
        if not sizes_only:
            for mat in matrices:
                print(mat)
                print()


def print_projective_geometry(q, n, d):
    points, blocks = projective_geometry_blocks(q, n, d, point_coordinates=True)
    print(f"PG({n},{q}) has {len(points)} point(s).")
    print(f"The d={d} projective subspaces give {len(blocks)} block(s).")
    if blocks:
        print("First block:")
        print(blocks[0])


def build_parser():
    parser = argparse.ArgumentParser(description="Grassmannian and finite-projective-geometry utilities for Sage Python.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    count = subparsers.add_parser("count", help="Count Gr(k,r)(F_q) and, for k=2, Schubert cell sizes.")
    count.add_argument("--q", type=int, default=2)
    count.add_argument("--r", type=int, default=4)
    count.add_argument("--k", type=int, default=2)

    schubert = subparsers.add_parser("schubert", help="Print Schubert cells for Gr(2,r)(F_q).")
    schubert.add_argument("--q", type=int, default=5)
    schubert.add_argument("--r", type=int, default=4)
    schubert.add_argument("--sizes", action="store_true", help="Only print cell sizes.")

    graph = subparsers.add_parser("graph", help="Plot the Sage Grassmann graph.")
    graph.add_argument("--q", type=int, default=2)
    graph.add_argument("--r", type=int, default=4)
    graph.add_argument("--k", type=int, default=2)
    graph.add_argument("--output", help="Output image path.")
    graph.add_argument("--no-show", action="store_true", help="Save or compute without opening a window.")

    subspace = subparsers.add_parser("subspace", help="Plot one Gr(2,r)(F_q) point on the GF(q^r) circle.")
    subspace.add_argument("--q", type=int, default=5)
    subspace.add_argument("--r", type=int, default=4)
    subspace.add_argument("--index", type=int, default=0)
    subspace.add_argument("--output", help="Output image path.")
    subspace.add_argument("--no-show", action="store_true", help="Save or compute without opening a window.")
    subspace.add_argument("--labels", action="store_true", help="Label highlighted alpha powers.")
    subspace.add_argument("--dpi", type=int, default=300)

    pg = subparsers.add_parser("pg-design", help="Print projective geometry incidence-design data.")
    pg.add_argument("--q", type=int, default=2)
    pg.add_argument("--n", type=int, default=3, help="Projective dimension.")
    pg.add_argument("--d", type=int, default=1, help="Projective subspace dimension.")

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "count":
        print_count(args.q, args.r, args.k)
    elif args.command == "schubert":
        print_schubert(args.q, args.r, sizes_only=args.sizes)
    elif args.command == "graph":
        graph = plot_grassmann_graph(args.q, args.r, args.k, args.output, show=not args.no_show)
        print(f"Grassmann graph J_{args.q}({args.r}, {args.k}): {graph.order()} vertices, {graph.size()} edges")
    elif args.command == "subspace":
        plot_2d_subspace_finite_field(
            args.q,
            r=args.r,
            index=args.index,
            output_filename=args.output,
            show=not args.no_show,
            labels=args.labels,
            dpi=args.dpi,
        )
    elif args.command == "pg-design":
        print_projective_geometry(args.q, args.n, args.d)


if __name__ == "__main__":
    main()
