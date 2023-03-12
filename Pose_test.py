import copy
import numpy as np
import open3d as o3d
import cv2
from scipy.optimize import least_squares
from utils import *
import matplotlib as plt
import time
import random
from scipy.spatial.transform import Rotation as R
import numpy as np

if __name__ == "__main__":

    # Intel RealSense D415
    depth_scaling_factor = 999.99
    focal_length = 597.522  ## mm
    img_center_x = 312.885
    img_center_y = 239.870

    # Set image path
    img_path=['./vatican/data/align_test{}.png'.format(i) for i in range(1,12)]
    depth_path=['./vatican/data/align_test_depth{}.png'.format(i) for i in range(1,12)]
    pcd=[o3d.io.read_point_cloud('./vatican/pcd/result{}.pcd'.format(i)) for i in range(1,12)]

    ########################################################################################################################
    # Feature matching using SIFT algorithm
    ########################################################################################################################
    # Find transformation matrix from corresponding points based on SIFT

    # Read image from path
    image=[cv2.imread(path) for path in img_path]
    depth=[np.array(o3d.io.read_image(path), np.float32) for path in depth_path]

    # Find keypoints and descriptors using SIFT
    sift=cv2.SIFT_create()
    sift_result=[(sift.detectAndCompute(img,None)) for img in image]
    boundary=[(get_boundary(p)) for p in pcd]


####################################################################################################################################################################

    result=pcd[0]
    #Get relative Pose
    for i in range(len(img_path)-1):
        relative_pose=SIFT_Transformation(img_path[i],img_path[i+1],depth_path[i],depth_path[i+1],pcd[i],pcd[i+1],distance_ratio=0.1)

        relative_pose=np.array(relative_pose)
        # Extract fpfh features
        pcd1_down, pcd1_fpfh = preprocess_point_cloud(pcd[i], 1e-2)
        pcd2_down, pcd2_fpfh = preprocess_point_cloud(pcd[i+1], 1e-2)
        registration1 = o3d.pipelines.registration.registration_icp(
            pcd1_down, pcd2_down, 1e-2, relative_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

        result+=result.transform(registration1.transformation)+pcd[i+1]
        o3d.visualization.draw_geometries([result])

