import cv2

def preprocess_image(raw_img, distortion_coeffs, K, params=None):

    img = raw_img.copy()
    if params is None:
        params = {
            'Equalize': False,      # Change to True if images are dark
            'Undistort': True,      # Usually True for real cameras
            'ClipLimit': 3.0 / 256,
            'NumTiles': (8, 8)
        }
    
    if params.get('Undistort', True):
        h, w = img.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, distortion_coeffs, (w, h), 1, (w, h))
        
        mapx, mapy = cv2.initUndistortRectifyMap(K, distortion_coeffs, None, new_K, (w, h), 5)
        img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    
    if params.get('Equalize', False):
        clahe = cv2.createCLAHE(
            clipLimit=params.get('ClipLimit', 3.0/256),
            tileGridSize=params.get('NumTiles', (8, 8))
        )
        img = clahe.apply(img)
    
    return img