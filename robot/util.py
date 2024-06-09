
import numpy as np
from numpy.typing import NDArray


def inverse_transformation_matrix(mat: NDArray) -> NDArray:
    rot = mat[:-1, :-1]
    trans = mat[:-1, -1]
    inv_mat = np.eye(len(mat))
    inv_mat[:-1, :-1] = rot.T
    inv_mat[:-1, -1] = -rot.T @ trans
    return inv_mat


def calculate_transformation_between_points(
        previous_points: NDArray, current_points: NDArray) -> NDArray:
    # calculate T (holds T.Current = Previous)
    # ref: https://www.jstage.jst.go.jp/article/jrsj/31/6/31_31_624/_pdf
    p_mean = np.mean(previous_points, axis=0)
    c_mean = np.mean(current_points, axis=0)
    p_minus_mean = previous_points - p_mean
    c_minus_mean = current_points - c_mean
    rot_mat = calculate_rotation_between_points(p_minus_mean, c_minus_mean)
    translation = p_mean - rot_mat @ c_mean
    n_dim = previous_points.shape[1]
    t_mat = np.eye(n_dim + 1)
    t_mat[:-1, :-1] = rot_mat
    t_mat[:-1, -1] = translation
    return t_mat


def calculate_rotation_between_points(previous_points: NDArray, current_points: NDArray) -> NDArray:
    # calculate R (holds R.Current = Previous)
    mat_x = previous_points.T @ current_points
    mat_u, _, mat_vt = np.linalg.svd(mat_x, full_matrices=True)
    if previous_points.shape[1] == 2:
        mat_s = np.diag([1., np.linalg.det(mat_u @ mat_vt)])
    else:
        mat_s = np.diag([1., 1., np.linalg.det(mat_u @ mat_vt)])
    rot_mat = mat_u @ mat_s @ mat_vt
    return rot_mat
