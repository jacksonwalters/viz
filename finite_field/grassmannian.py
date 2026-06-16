#!/usr/bin/env python3
"""Grassmannian utilities extracted from finite_field_visualization.ipynb.

Run with Sage's Python, for example:

    sage -python grassmannian.py count --q 2 --r 4
    sage -python grassmannian.py schubert --q 5 --r 4 --sizes
    sage -python grassmannian.py pg-lines --q 2 --output images/gr_2_4_pg_3_2.png
    sage -python grassmannian.py graph --q 2 --r 4 --k 2 --output images/grassmann_graph_2_4_2.png
    sage -python grassmannian.py subspace --q 5 --r 4 --index 0 --output images/gr_2_4_subspace_0.png
"""

import argparse
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sage.all import GF, I, VectorSpace, exp, graphs, matrix, pi, vector
from sage.matrix.echelon_matrix import reduced_echelon_matrix_iterator


SCHUBERT_COLORS = {
    (0, 1): "#4c78a8",
    (0, 2): "#f58518",
    (0, 3): "#54a24b",
    (1, 2): "#e45756",
    (1, 3): "#72b7b2",
    (2, 3): "#b279a2",
}


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


def field_value_indices(F):
    return {value: i for i, value in enumerate(F)}


def normalize_projective_vector(entries):
    entries = tuple(entries)
    for entry in entries:
        if entry != 0:
            scale = entry**-1
            return tuple(scale * value for value in entries)
    raise ValueError("the zero vector does not define a projective point")


def projective_sort_key(point, index_map):
    return tuple(index_map[value] for value in point)


def rref_pivot_columns(basis):
    pivots = []
    for row in basis.rows():
        for col, value in enumerate(row):
            if value != 0:
                pivots.append(col)
                break
    return tuple(pivots)


def projective_points(q, r=4):
    F = GF(q)
    points = []
    for point_matrix in reduced_echelon_matrix_iterator(F, 1, r, copy=True, set_immutable=True):
        points.append(normalize_projective_vector(point_matrix.row(0)))
    return points


def projective_points_on_subspace(q, basis):
    F = GF(q)
    zero = vector(F, [0] * basis.ncols())
    points = set()

    for coeffs in product(list(F), repeat=basis.nrows()):
        if all(coeff == 0 for coeff in coeffs):
            continue

        point = vector(F, zero)
        for coeff, row in zip(coeffs, basis.rows()):
            point += coeff * row
        points.add(normalize_projective_vector(point))

    index_map = field_value_indices(F)
    return sorted(points, key=lambda point: projective_sort_key(point, index_map))


def gr_2_4_projective_line_records(q):
    records = []
    for index, basis in enumerate(grassmannian_rref(q, 4, k=2)):
        records.append(
            {
                "index": index,
                "basis": basis,
                "cell": rref_pivot_columns(basis),
                "points": projective_points_on_subspace(q, basis),
            }
        )
    return records


def pg3_chart_coordinates(q, point, index_map):
    values = [index_map[value] for value in point]
    pivot = next(index for index, value in enumerate(point) if value != 0)

    if pivot == 0:
        a, b, c = values[1], values[2], values[3]
        return a + 0.48 * b, c + 0.38 * b, "x0=1"
    if pivot == 1:
        a, b = values[2], values[3]
        return q + 2.1 + a, b + 0.38 * (q - 1), "x0=0, x1=1"
    if pivot == 2:
        a = values[3]
        return 2 * q + 3.6, a + 0.38 * (q - 1), "x0=x1=0, x2=1"
    return 2 * q + 5.1, 0.38 * (q - 1), "x0=x1=x2=0, x3=1"


def draw_line_segments(ax, line_points, coordinates, color, alpha, linewidth, connect, zorder):
    if connect == "path":
        ordered_points = sorted(line_points, key=lambda point: (coordinates[point][0], coordinates[point][1]))
        pairs = zip(ordered_points, ordered_points[1:])
    else:
        pairs = combinations(line_points, 2)

    for p0, p1 in pairs:
        x0, y0 = coordinates[p0]
        x1, y1 = coordinates[p1]
        ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)


def parse_cell(cell_text):
    if cell_text is None:
        return None
    parts = [part.strip() for part in cell_text.split(",")]
    if len(parts) != 2:
        raise ValueError("cell must look like '0,1'")
    return tuple(int(part) for part in parts)


def plot_gr_2_4_as_pg_lines(
    q,
    output_filename=None,
    show=True,
    labels=False,
    highlight_index=None,
    cell_filter=None,
    connect="auto",
    line_alpha=None,
    line_width=1.0,
    dpi=300,
):
    """Visualize Gr(2,4)(F_q) as projective lines in PG(3,q)."""
    F = GF(q)
    index_map = field_value_indices(F)
    points = projective_points(q, r=4)
    records = gr_2_4_projective_line_records(q)
    coordinates = {
        point: pg3_chart_coordinates(q, point, index_map)[:2]
        for point in points
    }

    if cell_filter is not None:
        records_to_draw = [record for record in records if record["cell"] == cell_filter]
    else:
        records_to_draw = records

    if connect == "auto":
        base_connect = "complete" if q <= 2 else "path"
    else:
        base_connect = connect

    if line_alpha is None:
        line_alpha = 0.22 if q <= 2 else 0.06

    fig, ax = plt.subplots(figsize=(13, 7.5))

    selected_record = None
    if highlight_index is not None:
        selected_record = records[highlight_index % len(records)]

    for record in records_to_draw:
        color = SCHUBERT_COLORS.get(record["cell"], "#888888")
        alpha = line_alpha
        width = line_width
        zorder = 1

        if selected_record is not None:
            color = "#b8b8b8"
            alpha = min(alpha, 0.045 if q > 2 else 0.09)

        draw_line_segments(
            ax,
            record["points"],
            coordinates,
            color=color,
            alpha=alpha,
            linewidth=width,
            connect=base_connect,
            zorder=zorder,
        )

    if selected_record is not None:
        draw_line_segments(
            ax,
            selected_record["points"],
            coordinates,
            color="#d62728",
            alpha=0.95,
            linewidth=2.8,
            connect="complete",
            zorder=4,
        )

    x_values = [coordinates[point][0] for point in points]
    y_values = [coordinates[point][1] for point in points]
    ax.scatter(
        x_values,
        y_values,
        s=44 if q <= 2 else 30,
        color="#202020",
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )

    if selected_record is not None:
        selected_x = [coordinates[point][0] for point in selected_record["points"]]
        selected_y = [coordinates[point][1] for point in selected_record["points"]]
        ax.scatter(
            selected_x,
            selected_y,
            s=90 if q <= 2 else 72,
            color="#d62728",
            edgecolor="white",
            linewidth=1.0,
            zorder=6,
        )

    if labels:
        for point in points:
            x, y = coordinates[point]
            label = "(" + ",".join(str(index_map[value]) for value in point) + ")"
            ax.text(x + 0.05, y + 0.05, label, fontsize=7, zorder=7)

    chart_labels = {
        "first x0": (0.75 * (q - 1), -0.65),
        "first x1": (q + 2.1 + 0.5 * (q - 1), -0.65),
        "first x2": (2 * q + 3.6, -0.65),
        "first x3": (2 * q + 5.1, -0.65),
    }
    for label, (x, y) in chart_labels.items():
        ax.text(x, y, label, fontsize=9, ha="center", color="#4a4a4a")

    cell_counts = {}
    for record in records_to_draw:
        cell_counts[record["cell"]] = cell_counts.get(record["cell"], 0) + 1

    handles = [
        Line2D([0], [0], color=SCHUBERT_COLORS.get(cell, "#888888"), linewidth=2, label=f"{cell}: {count}")
        for cell, count in sorted(cell_counts.items())
    ]
    if selected_record is not None:
        handles.append(Line2D([0], [0], color="#d62728", linewidth=3, label=f"highlight {selected_record['index']}"))
    if handles:
        ax.legend(handles=handles, title="Schubert cells", loc="upper right", frameon=False)

    title = rf"$Gr(2,4)(\mathbb{{F}}_{q})$ as projective lines in $PG(3,{q})$"
    subtitle = f"{len(records)} lines, {len(points)} projective points"
    if cell_filter is not None:
        subtitle += f", cell {cell_filter}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=15)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min(x_values) - 0.8, max(x_values) + 1.2)
    ax.set_ylim(min(y_values) - 1.05, max(y_values) + 0.85)
    ax.axis("off")

    if output_filename:
        ensure_parent(output_filename)
        fig.savefig(output_filename, format=Path(output_filename).suffix.lstrip(".") or "png", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return records


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

    pg_lines = subparsers.add_parser("pg-lines", help="Visualize Gr(2,4)(F_q) as projective lines in PG(3,q).")
    pg_lines.add_argument("--q", type=int, default=2)
    pg_lines.add_argument("--output", help="Output image path.")
    pg_lines.add_argument("--no-show", action="store_true", help="Save or compute without opening a window.")
    pg_lines.add_argument("--labels", action="store_true", help="Label projective points by normalized coordinates.")
    pg_lines.add_argument("--highlight-index", type=int, help="Emphasize one Grassmannian element by index.")
    pg_lines.add_argument("--cell", help="Only draw one Schubert cell, for example '0,1'.")
    pg_lines.add_argument("--connect", choices=["auto", "path", "complete"], default="auto")
    pg_lines.add_argument("--alpha", type=float, help="Line opacity for non-highlighted projective lines.")
    pg_lines.add_argument("--line-width", type=float, default=1.0)
    pg_lines.add_argument("--dpi", type=int, default=300)

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
    elif args.command == "pg-lines":
        records = plot_gr_2_4_as_pg_lines(
            args.q,
            output_filename=args.output,
            show=not args.no_show,
            labels=args.labels,
            highlight_index=args.highlight_index,
            cell_filter=parse_cell(args.cell),
            connect=args.connect,
            line_alpha=args.alpha,
            line_width=args.line_width,
            dpi=args.dpi,
        )
        print(f"Gr(2,4)(F_{args.q}) as PG(3,{args.q}) lines: {len(records)} lines")
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
