"""Uniform-grid spatial index for fast neighbour queries.

The simulation calls into this on every tick (overlap detection,
divergent vacancy scan, erosion's k-NN). The grid is rebuilt from
scratch each call — cheaper than an incremental update for our N
(thousands), and the math is trivial.

The data structure is two arrays:

  - ``cell_start[c]`` — index into ``particle_order`` where cell ``c``'s
    member list begins.
  - ``particle_order[k]`` — particle index. Members of cell ``c`` are
    ``particle_order[cell_start[c]:cell_start[c+1]]``.

This is the standard "counting-sort by cell" representation: O(N) build,
no per-cell Python lists, all numpy. Queries return numpy arrays of
neighbour indices so callers stay vectorised.

Toroidal queries (``wrap=True``) treat the cell grid as a torus: halo
cells wrap modulo ``cols`` / ``rows``, and distance comparisons use the
toroidal shortest-path metric. The same data layout supports both modes
— wrap is a query-time option, set when the grid is built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tectonic_sim.types import WorldRect


@dataclass(frozen=True)
class BucketGrid:
    """A 2D uniform grid binning particles by position.

    Build once per tick via ``BucketGrid.build(...)``; query by
    ``neighbors_within(point, radius_km)`` for a single query or
    ``cross_label_pairs_within(labels, radius_km)`` for an all-pairs sweep.

    Attributes:
        domain: the world rectangle the grid covers.
        cell_size_km: side length of one cell.
        cols, rows: grid dimensions.
        positions: the ``(N, 2)`` particle positions this grid indexes.
        particle_order: ``(N,)`` int32 — particle indices grouped by cell.
        cell_start: ``(cols * rows + 1,)`` int32 — slice boundaries into
            ``particle_order``.
        wrap: when True, halo cells wrap modulo ``cols``/``rows`` and
            distance checks use the toroidal shortest-path metric.
    """

    domain: WorldRect
    cell_size_km: float
    cols: int
    rows: int
    positions: np.ndarray
    particle_order: np.ndarray
    cell_start: np.ndarray
    wrap: bool

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        positions: np.ndarray,
        domain: WorldRect,
        cell_size_km: float,
        *,
        wrap: bool = False,
    ) -> "BucketGrid":
        """Build a grid binning ``positions`` (in km) into uniform cells.

        ``cell_size_km`` should be on the order of the longest query
        radius — bigger cells mean fewer cells visited per query but
        more particles per cell.

        ``wrap`` selects toroidal queries (halo wraps; distance uses the
        toroidal shortest path). Default ``False`` keeps legacy
        rectangle-clipped behaviour for callers that haven't opted in.
        """
        if cell_size_km <= 0:
            raise ValueError(f"cell_size_km must be > 0, got {cell_size_km}")
        cols = max(1, int(math.ceil(domain.width_km / cell_size_km)))
        rows = max(1, int(math.ceil(domain.height_km / cell_size_km)))
        n_cells = cols * rows

        if positions.shape[0] == 0:
            return cls(
                domain=domain,
                cell_size_km=cell_size_km,
                cols=cols,
                rows=rows,
                positions=positions,
                particle_order=np.zeros(0, dtype=np.int32),
                cell_start=np.zeros(n_cells + 1, dtype=np.int32),
                wrap=wrap,
            )

        # Per-particle cell id. Out-of-domain particles get clamped to
        # the edge cell — callers should drop or wrap out-of-domain
        # particles before building, but the grid stays robust either way.
        ci = np.clip(
            ((positions[:, 0] + domain.half_width_km) / cell_size_km).astype(np.int32),
            0, cols - 1,
        )
        ri = np.clip(
            ((positions[:, 1] + domain.half_height_km) / cell_size_km).astype(np.int32),
            0, rows - 1,
        )
        cell_id = ci * rows + ri  # row-major within columns

        # Counting sort: histogram → cumulative → scatter.
        counts = np.bincount(cell_id, minlength=n_cells)
        cell_start = np.zeros(n_cells + 1, dtype=np.int32)
        cell_start[1:] = np.cumsum(counts)

        # Scatter particles into particle_order, in stable cell order.
        particle_order = np.argsort(cell_id, kind="stable").astype(np.int32)

        return cls(
            domain=domain,
            cell_size_km=cell_size_km,
            cols=cols,
            rows=rows,
            positions=positions,
            particle_order=particle_order,
            cell_start=cell_start,
            wrap=wrap,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _halo_cells(
        self, ci: int, ri: int, cell_halo: int,
    ) -> list[tuple[int, int]]:
        """Enumerate (cc, rr) halo cell indices around ``(ci, ri)``.

        Under wrap, indices are taken modulo ``cols`` / ``rows``; the
        halo extent is clamped to ``min(cols, rows) // 2`` so a single
        cell can't appear twice via both wrap directions on tiny grids.
        Under no-wrap, out-of-range cells are skipped.
        """
        out: list[tuple[int, int]] = []
        if self.wrap:
            max_halo_c = max(1, self.cols // 2)
            max_halo_r = max(1, self.rows // 2)
            halo_c = min(cell_halo, max_halo_c)
            halo_r = min(cell_halo, max_halo_r)
            for dci in range(-halo_c, halo_c + 1):
                cc = (ci + dci) % self.cols
                for dri in range(-halo_r, halo_r + 1):
                    rr = (ri + dri) % self.rows
                    out.append((cc, rr))
        else:
            for dci in range(-cell_halo, cell_halo + 1):
                cc = ci + dci
                if not (0 <= cc < self.cols):
                    continue
                for dri in range(-cell_halo, cell_halo + 1):
                    rr = ri + dri
                    if not (0 <= rr < self.rows):
                        continue
                    out.append((cc, rr))
        return out

    def _distance_squared_to_point(
        self, candidates: np.ndarray, point_km: tuple[float, float],
    ) -> np.ndarray:
        """Squared distance from each candidate particle to ``point_km``.

        Uses the toroidal metric when ``self.wrap`` is set.
        """
        dx = self.positions[candidates, 0] - point_km[0]
        dy = self.positions[candidates, 1] - point_km[1]
        if self.wrap:
            dx, dy = self.domain.wrapped_delta_xy(dx, dy)
        return dx * dx + dy * dy

    def _distance_squared_pair(
        self, ai: np.ndarray, bj: np.ndarray,
    ) -> np.ndarray:
        """Squared pair-wise distance between particle indices ``ai`` and
        ``bj``. Uses the toroidal metric when ``self.wrap`` is set."""
        dx = self.positions[ai, 0] - self.positions[bj, 0]
        dy = self.positions[ai, 1] - self.positions[bj, 1]
        if self.wrap:
            dx, dy = self.domain.wrapped_delta_xy(dx, dy)
        return dx * dx + dy * dy

    # -------------------------------------------------------------------------
    # Single-point query
    # -------------------------------------------------------------------------

    def neighbors_within(
        self, point_km: tuple[float, float], radius_km: float,
    ) -> np.ndarray:
        """All particle indices within ``radius_km`` of ``point_km``.

        Result is an ``int32`` array, unordered. Empty array if nothing
        is within range. Distance metric respects ``self.wrap``.
        """
        if self.positions.shape[0] == 0 or radius_km <= 0:
            return np.zeros(0, dtype=np.int32)

        px, py = point_km
        cell_halo = int(math.ceil(radius_km / self.cell_size_km))

        # Cell containing the query point. For wrap mode, mod into range
        # so an out-of-rectangle query point still hits the right cell.
        ci = int((px + self.domain.half_width_km) / self.cell_size_km)
        ri = int((py + self.domain.half_height_km) / self.cell_size_km)
        if self.wrap:
            ci %= self.cols
            ri %= self.rows
        else:
            ci = max(0, min(self.cols - 1, ci))
            ri = max(0, min(self.rows - 1, ri))

        # Gather candidate indices from the halo cells.
        chunks: list[np.ndarray] = []
        seen_cells: set[int] = set()
        for cc, rr in self._halo_cells(ci, ri, cell_halo):
            cid = cc * self.rows + rr
            if cid in seen_cells:
                continue
            seen_cells.add(cid)
            start = int(self.cell_start[cid])
            end = int(self.cell_start[cid + 1])
            if end > start:
                chunks.append(self.particle_order[start:end])
        if not chunks:
            return np.zeros(0, dtype=np.int32)
        candidates = np.concatenate(chunks)

        d2 = self._distance_squared_to_point(candidates, (px, py))
        return candidates[d2 <= radius_km * radius_km]

    # -------------------------------------------------------------------------
    # All-pairs sweep (used by overlap detection)
    # -------------------------------------------------------------------------

    def cross_label_pairs_within(
        self, labels: np.ndarray, radius_km: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """All unordered pairs of particle indices with *different* labels
        and pairwise distance ≤ ``radius_km``.

        Each unordered pair appears exactly once in the output, but the
        index ordering within a pair (``i < j`` or ``j < i``) is *not*
        guaranteed — only that ``labels[i] != labels[j]`` and the two
        particles are within ``radius_km`` (under the toroidal metric
        when ``self.wrap`` is set).
        """
        return self._labelled_pairs_within(labels, radius_km, cross=True)

    def same_label_pairs_within(
        self, labels: np.ndarray, radius_km: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """All unordered pairs of particle indices with the *same* label
        and pairwise distance ≤ ``radius_km``.

        Mirrors ``cross_label_pairs_within`` with the label predicate
        flipped. Used by the intra-plate spacing constraint to detect
        same-plate neighbours that have been compressed below the rest
        spacing by cross-plate contact pressure.
        """
        return self._labelled_pairs_within(labels, radius_km, cross=False)

    def _labelled_pairs_within(
        self, labels: np.ndarray, radius_km: float, *, cross: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shared cell-sweep for cross-label and same-label pair queries.

        ``cross=True`` keeps pairs with differing labels; ``cross=False``
        keeps pairs with matching labels. The cell walk, halo geometry,
        forward-only dedup, and distance filter are identical — only the
        label-comparison direction in ``_emit_pairs`` differs.
        """
        if self.positions.shape[0] == 0 or radius_km <= 0:
            empty = np.zeros(0, dtype=np.int32)
            return empty, empty

        cell_halo = int(math.ceil(radius_km / self.cell_size_km))
        r2 = radius_km * radius_km

        # Walk every cell and pair its members against the halo cells'
        # members. Dedup pair-of-cells visits via a "forward-only" offset
        # rule that works for both wrap and no-wrap modes:
        #   for cross-cell pairs, include only (dci > 0) or
        #   (dci == 0 and dri > 0). The within-cell case is handled by
        #   triu_indices.
        if self.wrap:
            max_halo_c = max(1, self.cols // 2)
            max_halo_r = max(1, self.rows // 2)
            halo_c = min(cell_halo, max_halo_c)
            halo_r = min(cell_halo, max_halo_r)
        else:
            halo_c = halo_r = cell_halo

        out_i: list[np.ndarray] = []
        out_j: list[np.ndarray] = []

        for ci in range(self.cols):
            for ri in range(self.rows):
                cid = ci * self.rows + ri
                a_start = int(self.cell_start[cid])
                a_end = int(self.cell_start[cid + 1])
                if a_end == a_start:
                    continue
                a_idx = self.particle_order[a_start:a_end]

                for dci in range(-halo_c, halo_c + 1):
                    for dri in range(-halo_r, halo_r + 1):
                        # Within-cell.
                        if dci == 0 and dri == 0:
                            if len(a_idx) < 2:
                                continue
                            ii, jj = np.triu_indices(len(a_idx), k=1)
                            ai = a_idx[ii]
                            bj = a_idx[jj]
                            self._emit_pairs(
                                ai, bj, labels, r2, out_i, out_j, cross=cross,
                            )
                            continue

                        # Forward-only offsets to dedup cell-cell pairs.
                        if dci < 0 or (dci == 0 and dri <= 0):
                            continue

                        if self.wrap:
                            cc = (ci + dci) % self.cols
                            rr = (ri + dri) % self.rows
                        else:
                            cc = ci + dci
                            rr = ri + dri
                            if not (0 <= cc < self.cols and 0 <= rr < self.rows):
                                continue
                        nid = cc * self.rows + rr
                        # Wrap-induced self-pair (rare): if the halo cell
                        # is the same cell, skip — within-cell already
                        # handled it.
                        if nid == cid:
                            continue
                        b_start = int(self.cell_start[nid])
                        b_end = int(self.cell_start[nid + 1])
                        if b_end == b_start:
                            continue
                        b_idx = self.particle_order[b_start:b_end]

                        ai = np.repeat(a_idx, len(b_idx))
                        bj = np.tile(b_idx, len(a_idx))
                        self._emit_pairs(
                            ai, bj, labels, r2, out_i, out_j, cross=cross,
                        )

        if not out_i:
            empty = np.zeros(0, dtype=np.int32)
            return empty, empty
        return (
            np.concatenate(out_i).astype(np.int32, copy=False),
            np.concatenate(out_j).astype(np.int32, copy=False),
        )

    def _emit_pairs(
        self,
        ai: np.ndarray,
        bj: np.ndarray,
        labels: np.ndarray,
        r2: float,
        out_i: list[np.ndarray],
        out_j: list[np.ndarray],
        *,
        cross: bool,
    ) -> None:
        """Apply the label predicate + distance filter; append survivors.

        ``cross=True`` keeps ``labels[ai] != labels[bj]``;
        ``cross=False`` keeps ``labels[ai] == labels[bj]``.
        """
        if ai.size == 0:
            return
        keep = (labels[ai] != labels[bj]) if cross else (labels[ai] == labels[bj])
        if not keep.any():
            return
        ai = ai[keep]
        bj = bj[keep]
        d2 = self._distance_squared_pair(ai, bj)
        mask = d2 <= r2
        if not mask.any():
            return
        out_i.append(ai[mask])
        out_j.append(bj[mask])
