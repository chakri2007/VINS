import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from .fast_detector import FastDetector
from .klt_tracker import KltTracker
from .gridder import ImageGridder, DeficitCell


class FeatureExtractor:
    def __init__(self, frame_size: Tuple[int, int] = (640, 480)):
        self.fast_detector = FastDetector()
        self.klt_tracker = KltTracker(max_level=3, win_size=(21, 21))
        self.gridder = ImageGridder(
            frame_size=frame_size,
            cell_rows=5,
            cell_cols=5,
            min_features_per_cell=8,
            border_pad=5,
        )

        self._cell_thread_pool = ThreadPoolExecutor(
            max_workers=self.gridder.total_cells,
            thread_name_prefix="fast_cell",
        )

    def detect_initial_features(self, gray_frame: np.ndarray) -> np.ndarray:
        return self.fast_detector.detect(gray_frame)
    
    def track_features(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.klt_tracker.track(prev_frame, curr_frame, prev_points)


    def detect_new_features(
        self,
        gray_frame: np.ndarray,
        existing_points: np.ndarray,
    ) -> np.ndarray:
        mask = np.ones(gray_frame.shape, dtype=np.uint8) * 255
        for pt in existing_points:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), radius=10, color=0, thickness=-1)

        return self.fast_detector.detect(gray_frame, mask=mask)

    def extract_features_in_empty_cells(
        self,
        gray_frame: np.ndarray,
        tracked_points: np.ndarray,  # shape (N, 2) — current frame positions
    ) -> np.ndarray:
        deficit_cells: List[DeficitCell] = self.gridder.get_deficit_cells(tracked_points)

        if not deficit_cells:
            return np.empty((0, 2), dtype=np.float32)

        full_mask = self._build_exclusion_mask(gray_frame.shape, tracked_points, radius=10)

        # Submit one job per deficit cell to the thread pool.
        futures = {
            self._cell_thread_pool.submit(
                self._detect_in_cell,
                gray_frame,
                full_mask,
                cell,
            ): cell
            for cell in deficit_cells
        }

        all_new_points: List[np.ndarray] = []

        for future in as_completed(futures):
            cell_points = future.result()  # shape (k, 2) in full-frame coords
            if cell_points is not None and len(cell_points) > 0:
                all_new_points.append(cell_points)

        if not all_new_points:
            return np.empty((0, 2), dtype=np.float32)

        return np.vstack(all_new_points).astype(np.float32)


    def _detect_in_cell(
        self,
        gray_frame: np.ndarray,
        full_mask: np.ndarray,
        cell: DeficitCell,
    ) -> Optional[np.ndarray]:
        x, y, w, h = cell.roi

        cell_frame = gray_frame[y:y + h, x:x + w]
        cell_mask  = full_mask[y:y + h, x:x + w]

        cell_points = self.fast_detector.detect(cell_frame, mask=cell_mask)

        if cell_points is None or len(cell_points) == 0:
            return None

        cell_points[:, 0] += x
        cell_points[:, 1] += y

        if len(cell_points) > cell.deficit:
            cell_points = cell_points[:cell.deficit]

        return cell_points

    @staticmethod
    def _build_exclusion_mask(
        frame_shape: Tuple[int, ...],
        existing_points: np.ndarray,
        radius: int = 10,
    ) -> np.ndarray:
        mask = np.full(frame_shape[:2], 255, dtype=np.uint8)
        if existing_points is not None and len(existing_points) > 0:
            for pt in existing_points:
                cv2.circle(mask, (int(pt[0]), int(pt[1])), radius, 0, -1)
        return mask

    def shutdown(self):
        self._cell_thread_pool.shutdown(wait=False)