import numpy as np

import preprocess_image

from feature_manager.feature_extractor import FeatureExtractor
from img_view_set import ImageViewSet

class VisualInertialOdometry():
    def __init__(self, calib_data):

        self.left_calib        = calib_data['left']
        self.intrinsics        = self.left_calib['intrinsics']
        self.distortion_coeffs = np.array(
            self.left_calib['distortion_coefficients']
        )
        self.T_BS = np.array(
            self.left_calib['T_BS']['data']
        ).reshape(4, 4)

        self.K = np.array([
            [self.intrinsics[0], 0,                  self.intrinsics[2]],
            [0,                  self.intrinsics[1], self.intrinsics[3]],
            [0,                  0,                  1],
        ], dtype=np.float64)

        # Processing parameters
        self.params = {
            'Equalize': False,      
            'Undistort': True,    
            'ClipLimit': 3.0 / 256,
            'NumTiles': (8, 8)
        }
        
        self.feature_manager = FeatureManager()
        self.feature_extractor = FeatureExtractor()
        self.view_set = ImageViewSet()

        self.isFirstFrame = True
        self.isFirstFewFrames = True
        self.isMapInitialized = False

        self.isVIO_initialized = False
        self.isVI_aligned = False

        self.frameID = 0

        
    def vio_loop(self,raw_img_frame, img_frame_timestamp):
        self.frameID += 1
        self.img_frame = preprocess_image(
            raw_img_frame, 
            self.distortion_coeffs, 
            self.K, 
            self.params
        )

        if not self.isVIO_initialized:
            self.vio_initialization(self.img_frame, img_frame_timestamp, self.frameID)
        elif not self.isVI_aligned :
            self.VI_alignment()
        else :
            self.visual_inertial_optimization()




    #PHASE - 1

    def vio_initialization(self,img_frame, img_frame_timestamp, frameID):

        if self.isFirstFrame :

            current_features = self.feature_extractor.detect_initial_features(img_frame)


            update_sliding_window(img_frame, current_features, frameID)
            #add frame to sliding window

            self.view_set.add_view(view_id=self.frameID, pose=None)

            self.first_img_frame = img_frame
            self.isFirstFrame = False
        else:
            #track detected features
            #remove lost features
            #detect new features

            if self.isFirstFewFrames :
                #add the frame to sliding window
                self.isFirstFewFrames = False
                
            elif self.window.enough_parallax :
                #RANSAC
                #Recover pose
                #Store pose and frame in memory management
                self.isMapInitialized = True

            

    #PHASE - 2

    def VI_alignment():
        pass

    #PHASE - 3

    def visual_inertial_optimization():
        pass