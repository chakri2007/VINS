import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DeficitCell:
    cell_row: int               # grid row index
    cell_col: int               # grid col index
    cell_id: int                # flat index = cell_row * n_cols + cell_col
    roi: Tuple[int, int, int, int]  # (x, y, w, h) in full-frame pixel coords
    deficit: int                # how many more features this cell wants


class ImageGridder:
    def __init__(
        self,
        frame_size: Tuple[int, int],
        cell_rows: int = 4,
        cell_cols: int = 6,
        min_features_per_cell: int = 3,
        border_pad: int = 5,
    ):
        self.frame_w, self.frame_h = frame_size
        self.cell_rows = cell_rows
        self.cell_cols = cell_cols
        self.min_features_per_cell = min_features_per_cell
        self.border_pad = border_pad

        self._x_edges = np.linspace(0, self.frame_w, cell_cols + 1, dtype=np.int32)
        self._y_edges = np.linspace(0, self.frame_h, cell_rows + 1, dtype=np.int32)

        self.total_cells = cell_rows * cell_cols


    def get_deficit_cells(
        self,
        tracked_points: np.ndarray,  # shape (N, 2), float32, [x, y]
    ) -> List[DeficitCell]:
        counts = self._count_points_per_cell(tracked_points)

        deficit_cells = []
        for r in range(self.cell_rows):
            for c in range(self.cell_cols):
                cell_id = r * self.cell_cols + c
                count   = counts[r, c]
                if count < self.min_features_per_cell:
                    roi     = self._cell_roi(r, c)
                    deficit = self.min_features_per_cell - count
                    deficit_cells.append(DeficitCell(
                        cell_row=r,
                        cell_col=c,
                        cell_id=cell_id,
                        roi=roi,
                        deficit=deficit,
                    ))

        return deficit_cells

    def get_cell_occupancy(self, tracked_points: np.ndarray) -> np.ndarray:
        return self._count_points_per_cell(tracked_points)


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_points_per_cell(self, tracked_points: np.ndarray) -> np.ndarray:
        counts = np.zeros((self.cell_rows, self.cell_cols), dtype=np.int32)

        if tracked_points is None or len(tracked_points) == 0:
            return counts

        pts = np.asarray(tracked_points, dtype=np.float32)
        xs  = pts[:, 0]
        ys  = pts[:, 1]

        # digitize returns bin index in [1, n_cells], clip to valid range
        col_idx = np.digitize(xs, self._x_edges[1:])   # 0 .. cell_cols-1
        row_idx = np.digitize(ys, self._y_edges[1:])   # 0 .. cell_rows-1

        col_idx = np.clip(col_idx, 0, self.cell_cols - 1)
        row_idx = np.clip(row_idx, 0, self.cell_rows - 1)

        # Accumulate — np.add.at handles duplicate indices correctly
        np.add.at(counts, (row_idx, col_idx), 1)

        return counts

    def _cell_roi(self, r: int, c: int, padded: bool = True) -> Tuple[int, int, int, int]:
        x1 = int(self._x_edges[c])
        x2 = int(self._x_edges[c + 1])
        y1 = int(self._y_edges[r])
        y2 = int(self._y_edges[r + 1])

        if padded and self.border_pad > 0:
            x1 = max(0, x1 - self.border_pad)
            y1 = max(0, y1 - self.border_pad)
            x2 = min(self.frame_w, x2 + self.border_pad)
            y2 = min(self.frame_h, y2 + self.border_pad)

        return (x1, y1, x2 - x1, y2 - y1)

    def get_overcrowded_evictions(
        self,
        tracked_points: np.ndarray,   # shape (N, 2)
        track_ids: np.ndarray,        # shape (N,)
        track_ages: np.ndarray,       # shape (N,)
        max_features_per_cell: int = None,
    ) -> np.ndarray:
        """
        Returns IDs of tracks to evict from over-occupied cells.
        Eviction policy: keep oldest tracks (longest baseline), drop youngest.
        """
        if max_features_per_cell is None:
            max_features_per_cell = self.min_features_per_cell * 2  # 2x headroom

        if len(tracked_points) == 0:
            return np.empty((0,), dtype=np.int64)

        pts = np.asarray(tracked_points, dtype=np.float32)
        col_idx = np.clip(np.digitize(pts[:, 0], self._x_edges[1:]), 0, self.cell_cols - 1)
        row_idx = np.clip(np.digitize(pts[:, 1], self._y_edges[1:]), 0, self.cell_rows - 1)

        evict_ids = []
        for r in range(self.cell_rows):
            for c in range(self.cell_cols):
                mask = (row_idx == r) & (col_idx == c)
                cell_indices = np.where(mask)[0]

                if len(cell_indices) <= max_features_per_cell:
                    continue

                # Sort by age descending — keep oldest, evict youngest
                ages_in_cell = track_ages[cell_indices]
                sorted_order = np.argsort(-ages_in_cell)   # descending age
                evict_local = sorted_order[max_features_per_cell:]
                evict_ids.extend(track_ids[cell_indices[evict_local]].tolist())

        return np.array(evict_ids, dtype=np.int64)