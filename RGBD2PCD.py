import argparse
import sys
import os
import numpy as np
import open3d as o3d
from PIL import Image

focalLength = 597.522
centerX = 312.885
centerY = 239.870
scalingFactor=1000.0

for i in range(1,12):
    depth=Image.open("./vatican/data/align_test_depth{}.png".format(i))
    rgb=Image.open("./vatican/data/align_test{}.png".format(i))



    colors = []
    points = []
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u, v))
            Z = depth.getpixel((u, v)) / scalingFactor
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            if Z>1.5:
                continue
            points.append([X,Y,Z])
            colors.append([color[0]/255.0,color[1]/255.0,color[2]/255.0])

    points=np.array(points)
    colors=np.array(colors)

    # Convert to Open3D.PointCLoud:
    pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(points)
    pcd_o3d.colors=o3d.utility.Vector3dVector(colors)

    # Visualize:
    o3d.visualization.draw_geometries([pcd_o3d])
    o3d.io.write_point_cloud("./desk2/pcd/result{}.pcd".format(i),pcd_o3d)
