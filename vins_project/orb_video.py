import cv2 as cv
import numpy as np
import math

def orb_features():
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    while (cap.isOpened()):
        ret , frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        gray = cv.drawKeypoints(
    gray,
    keypoints,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
        cv.imshow('ORB Features', gray)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


def grid_FAST(total_features, grid_x, grid_y):
    if (total_features < grid_x * grid_y):
        print("Total features must be greater than grid_x * grid_y")
        ratio = float(grid_x/grid_y)
        grid_y = math.ceil(math.sqrt(total_features/ratio))
        grid_x = math.ceil(ratio * grid_y)

    num_features_per_cell = int(total_features / (grid_x * grid_y)) + 1
    return num_features_per_cell


if __name__ == "__main__":
    orb_features()