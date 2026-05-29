import cv2 as cv
import math
from joblib import Parallel, delayed

keypoints = []
empty_cells = []

def build_img_pyramid(img, scale_factor,num_levels):
    pyramid = [img]

    for i in range(1, num_levels):
        new_w = max(1,int(pyramid[i-1].shape[1] / scale_factor))
        new_h = max(1,int(pyramid[i-1].shape[0]/ scale_factor))

        pyramid.append(cv.resize(pyramid[i-1], (new_w, new_h), interpolation=cv.INTER_LINEAR))
    return pyramid


def compute_grid_size(total_features, img_width, img_height):
    grids = []
    j = math.ceil(math.sqrt(total_features * img_width / img_height))  # cols
    i = math.ceil(math.sqrt(total_features * img_height / img_width))  # rows
    rows = max(1, i)
    cols = max(1, j)
    return rows, cols

def build_grids(img, grid_x, grid_y):
    h, w = img.shape[:2]
    cell_w = w // grid_x
    cell_h = h // grid_y
    for i in range(grid_y):
        for j in range(grid_x):
            x_start = j * cell_w
            y_start = i * cell_h
            x_end = min(x_start + cell_w, w)
            y_end = min(y_start + cell_h, h)
            grids.append((x_start, y_start, x_end, y_end))
    return grids


def _detect_in_cell(img, grid,fast_threshold):

    # Create detector inside the function — safe for parallel workers
    fast = cv.FastFeatureDetector_create(
        threshold=fast_threshold,
        nonmaxSuppression=True
    )
    x_start, y_start, x_end, y_end = grid
    kps = fast.detect(img[y_start:y_end, x_start:x_end], None)

    if not kps:
        compensate_empty_cell(img, grid, fast_threshold)

    # Keep best corner by response
    best = max(kps, key=lambda k: k.response)

    # CRITICAL: shift local ROI coords → full image coords
    best.pt = (best.pt[0] + x_start, best.pt[1] + y_start)

    return best


def grid_fast_parallel(grids,img , fast_threshold):

    # n_jobs=-1 uses all cores. prefer='threads' avoids pickling the image
    # (threads share memory — numpy arrays are safe with thread backend)
    keypoints = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_detect_in_cell)(img, grid, fast_threshold)
            for grid in grids
        )
    
    for i in range(len(keypoints)):
        if keypoints[i] is None:
            empty_cells.append(grids[i])

    
    return keypoints 

def compensate_empty_cell(img, grid, fast_threshold):
    found = False
    radius  = 0
    while not found:
        radius += 1
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if abs(x) == radius or abs(y) == radius:  # Check only the border of the square
                    new_grid = (grid[0] + x, grid[1] + y, grid[2] + x, grid[3] + y)
                    if (0 <= new_grid[0] < img.shape[1] and 0 <= new_grid[1] < img.shape[0] and
                        0 <= new_grid[2] < img.shape[1] and 0 <= new_grid[3] < img.shape[0]):
                        kps = _detect_in_cell(img, new_grid, fast_threshold)
                        if kps:
                            found = True
                            break
            if found:
                break
