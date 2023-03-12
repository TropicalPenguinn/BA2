from open3d import *
import numpy as np
import cv2
import open3d as o3d
from plot import draw_registration_result


def match_ransac(p, p_prime, tol = 0.01):
    """
    A ransac process that estimates the transform between two set of points p and p_prime.
    The transform is returned if the RMSE of the smallest 70% is smaller than the tol.

    Parameters
    ----------
    source_pcd : (n,3) float
      The source 3d pointcloud as a numpy.ndarray
    target_pcd : (n,3) float
      The target 3d pointcloud as a numpy.ndarray
    tol : float
      A transform is considered found if the smallest 70% RMSE error between the
      transformed p to p_prime is smaller than the tol
    Returns
    ----------
    transform: (4,4) float or None
      The homogeneous rigid transformation that transforms p to the p_prime's
      frame
      if None, the ransac does not find a sufficiently good solution

    """

    leastError = None
    R = None
    t= None
    # the smallest 70% of the error is used to compute RMSE
    k= int(len(p)*0.7)
    assert len(p) == len(p_prime)
    R_temp,t_temp = rigid_transform_3D(p,p_prime)
    R_temp = np.array(R_temp)
    t_temp = (np.array(t_temp).T)[0]
    transformed = (np.dot(R_temp, p.T).T)+t_temp
    error = (transformed - p_prime)**2
    error = np.sum(error, axis=1)
    error = np.sqrt(error)

    RMSE = np.sum(error[np.argpartition(error, k)[:k]])/k
    if RMSE < tol:
        R = R_temp
        t = t_temp

        transform = [[R[0][0],R[0][1],R[0][2],t[0]],
                     [R[1][0],R[1][1],R[1][2],t[1]],
                     [R[2][0],R[2][1],R[2][2],t[2]],
                     [0,0,0,1]]
        return transform
    else:
        return None

def rigid_transform_3D(A, B):
    """
    Estimate a rigid transform between 2 set of points of equal length
    through singular value decomposition(svd), return a rotation and a
    transformation matrix
    Parameters
    ----------
    A : (n,3) float
      The source 3d pointcloud as a numpy.ndarray
    B : (n,3) float
      The target 3d pointcloud as a numpy.ndarray
    Returns
    ----------
    R: (3,3) float
      A rigid rotation matrix
    t: (3) float
      A translation vector

    """

    assert len(A) == len(B)
    A=  np.asmatrix(A)
    B=  np.asmatrix(B)
    N = A.shape[0];

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = AA.T * BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    # reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    return (R, t)

