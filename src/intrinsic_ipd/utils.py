from .constants import CAD_FILES, DATASET_IDS, CAMERA_NAMES

import os
import urllib.request
import zipfile
from tqdm import tqdm
from typing import Optional, Union
import numpy as np
import itertools
from scipy.spatial.transform import Rotation as R


import logging

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download(url, to_dir: Union[str, os.PathLike]) -> Optional[Union[str, os.PathLike]]:
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    file_path = os.path.join(to_dir, url.split("/")[-1])
    if not os.path.exists(file_path):
        try:
            with DownloadProgressBar(unit='B', unit_scale=True,
                                miniters=1, desc=url.split('/')[-1]) as t:
                file_path, _ = urllib.request.urlretrieve(url, filename=file_path, reporthook=t.update_to)
        except Exception as e:
            os.remove(file_path)
            raise e
    else:
        logging.debug(f"{file_path} already exists")
        return
    return file_path

def extract(zip_path, to_dir: Union[str, os.PathLike]) -> None:
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=to_dir)
    print(f"Extracted {zip_path} to {to_dir}")

def download_cads(to_dir: Union[str, os.PathLike]) -> None:
    for cad_name in CAD_FILES:
        url = f"https://storage.googleapis.com/akasha-public/industrial_plenoptic_dataset/cad_models/{cad_name}.stl"
        download(url, to_dir=f"{to_dir}/models")
        # TODO
        # url = f"https://storage.googleapis.com/akasha-public/industrial_plenoptic_dataset/cad_models/{cad_name}_symm.json"
        # download(url, to_dir=f"{to_dir}/models")

def download_dataset(dataset_id : str, camera_name : str, to_dir : Union[str, os.PathLike]) -> Optional[Union[str, os.PathLike]]:
    assert dataset_id in DATASET_IDS, f"Invalid dataset id {dataset_id}, must be one of {DATASET_IDS}"
    assert camera_name in CAMERA_NAMES, f"Invalid camera name {camera_name}, must be one of {CAMERA_NAMES}"
    dataset_name = f'{camera_name}-{dataset_id}'
    url = f"https://storage.googleapis.com/akasha-public/industrial_plenoptic_dataset/{dataset_name}.zip"
    zip_path = download(url, to_dir=to_dir)
    return zip_path

def verify_symmetry(symm_params):
    """
    Examines a set of symmetry parameters and returns
    if they're valid.

    Args:
        symm_params (dict): See `vectorized_remove_symmetry` for in-depth description.

    Returns:
        Tuple(bool, str or None):
            - True if symmetry parameters are valid
            - Error message if symmetry parameters are not valid
    """

    num_cont = 0
    discrete_symmetries = []
    for axis in "xyz":
        if isinstance(symm_params.get(axis, None), dict):
            if symm_params[axis]["mod"] == 0:
                num_cont += 1
            else:
                discrete_symmetries.append(symm_params[axis]["mod"])

                if 360 % symm_params[axis]["mod"] != 0:
                    return False, "Discrete mod symmetry values must divide 360 evenly."
    if num_cont == 2:
        return False, "There can only be 0, 1, or 3 continuous symmetry axes on an object."

    if num_cont == 1:
        if len(set(discrete_symmetries).difference([180])) > 0:
            return False, "Only 180 degree symmetries are allowed for the other axes" \
                          " when a continuous symmetric axis is present."

    return True, None

def extract_symmetry_params(x=None, y=None, z=None, ref_pose=None):
    result, message = verify_symmetry(dict(x=x, y=y, z=z))
    if not result:
        raise ValueError(f"Invalid Symmetry x={x}, y={y}, z={z}: " + message)

    symm_dims = [np.eye(3)]

    cont_symm_axes = []
    discrete_symms = dict(X=[0], Y=[0], Z=[0])

    if x is not None:
        if x["mod"] == 0:
            cont_symm_axes.append(0)
        else:
            for i in np.arange(0, 360, x["mod"]):
                if i == 0:
                    continue
                symm_dims.append(R.from_euler("xyz", [np.radians(i), 0, 0]).as_matrix())

            discrete_symms["X"] = np.arange(0, 360, x["mod"])

    if y is not None:
        if y["mod"] == 0:
            cont_symm_axes.append(1)
        else:
            for i in np.arange(0, 360, y["mod"]):
                if i == 0:
                    continue
                symm_dims.append(R.from_euler("yxz", [np.radians(i), 0, 0]).as_matrix())

            discrete_symms["Y"] = np.arange(0, 360, y["mod"])

    if z is not None:
        if z["mod"] == 0:
            cont_symm_axes.append(2)
        else:
            for i in np.arange(0, 360, z["mod"]):
                if i == 0:
                    continue
                symm_dims.append(R.from_euler("zyx", [np.radians(i), 0, 0]).as_matrix())

            discrete_symms["Z"] = np.arange(0, 360, z["mod"])

    proper_symms = []
    for axes in itertools.permutations("XYZ"):
        axes = "".join(axes)
        eulers = list(itertools.product(
            discrete_symms[axes[0]],
            discrete_symms[axes[1]],
            discrete_symms[axes[2]]
        ))
        proper_symms += [
            tuple(rotvec) for rotvec in R.from_euler(axes, eulers, degrees=True).as_rotvec()
        ]

    proper_symms = R.from_rotvec(proper_symms).as_matrix()
    # proper_symms may have duplicates, remove them
    diff_mat = np.linalg.norm(proper_symms[None] - proper_symms[:, None], axis=(-1, -2))
    proper_symms = proper_symms[[i for i, row in enumerate(diff_mat)
                                 if i == 0 or all(row[:i] > 1e-9)]]

    if ref_pose is None:
        ref_pose = np.eye(3)

    symmetry_mode = "full_discrete"
    if len(cont_symm_axes) > 1:
        symmetry_mode = "full_continuous"
    elif len(cont_symm_axes) == 1:
        symmetry_mode = "semi_continuous"

    params = dict(
        discrete_symm_rots=symm_dims,
        proper_symms=np.array(proper_symms),
        symmetry_mode=symmetry_mode,
        continuous_symm_axis=cont_symm_axes[-1] if len(cont_symm_axes) > 0 else -1,
        fix_continuous_symm_angles=len(cont_symm_axes) == 0,
        ref_pose=np.array(ref_pose)
    )

    return params

def skew_symmetric_3(v):
    """
    Given a set of vectors v (Nx3), this function returns
    Nx3x3, the skew symmetric matrix of the vectors.
    """

    zero_arr = np.zeros(len(v))

    return np.transpose(np.array([
        [zero_arr, -v[:, 2], v[:, 1]],
        [v[:, 2], zero_arr, -v[:, 0]],
        [-v[:, 1], v[:, 0], zero_arr]
    ]), (2, 0, 1))

def vectorized_rot_between(vec1, vec2):
    """
    Given a set of normalized vectors, we compute the rotation matrix between
    the first set to the second set.

    Args:
        vec1 (np.array): Array of size Nx3
        vec2 (np.array): Array of size Nx3

    Returns:
        Rotation matrices R such that R[i] @ vec1[i] = vec2[i]
    """
    cosine = np.sum((vec1 * vec2), axis=1)

    result = np.zeros((len(cosine), 3, 3))
    result[cosine >= 1] = np.eye(3)
    result[cosine <= -1] = R.from_rotvec([0, np.pi, 0]).as_matrix()

    to_compute = np.where(np.abs(cosine) != 1)

    v = np.cross(vec1[to_compute], vec2[to_compute])
    v_x = skew_symmetric_3(v)

    result[to_compute] = np.eye(3)[None, :] + v_x + (v_x @ v_x) / (1.0 + cosine[to_compute][:, None, None])

    return result

def vectorized_remove_symmetry(poses, symm_params, inplace=False):
    """
    Given a set of symmetry parameters (see utils.extract_symmetry_params) and input poses,
    this function reduces all poses that are considered rotationally symmetric with
    each other to the same pose. This is mostly important for 6DOF pose estimation.

    In general, the resulting pose is as close as possible to the "ref_pose" value
    (called the "reference pose") in the symmetry parameters, while still being
    symmetric to the input pose.

    There are three cases:
    1) An object has three continuous symmetry axes (i.e. it's a sphere)

        In this case, we reduce the poses to the reference pose.

    2) An object has one axis of continuous symmetry and up to 2 other
        discrete symmetries (which must be 180 degrees, if symmetric)

        If discrete symmetries exist, we flip the continuous axis to
        be as close as possible to the reference pose's corresponding axis.
        We then adjust the reference pose so that its corresponding axis
        aligns with the flipped or non-flipped continuous axis.

    3) An object has *only* discrete symmetries.

        In this case, we iterate through all symmetric transformations
        (which are encoded in the symmetry_parameters) and pick the
        one that transforms the input pose as close as possible
        to the reference pose (we use the Frobenius norm of the
        difference between the two poses).

    Args:
        poses (np.array): Nx4x4 NumPy array, pose to be reduced
        symm_params (dict): dict(
            proper_symms (np.array): Nx4x4, all discrete transformations
                between symmetric poses
            ref_pose (np.array): 4x4, the reference pose to reduce to
                (or to reduce close to)
            continuous_symm_axis (int): axis of continuous symmetry,
                if only one exists
            symmetry_mode (str): Is one of the following:
                "full_continuous": All three axes have continuous symmetry
                "semi_continuous": One axis has continuous symmetry
                "full_discrete": No axes have continuous symmetry
        )
        inplace (bool): If true, directly alters the passed in input pose.
                        Defaults to True.

    Returns:
        np.array: The symmetry reduced pose

    """

    if not inplace:
        poses = poses.copy().astype(np.float128)

    ref_pose = symm_params["ref_pose"]

    symm_reductions = symm_params["proper_symms"]

    if symm_params["symmetry_mode"] == "full_continuous":
        poses[:, :3, :3] = symm_params["ref_pose"]
    elif symm_params["symmetry_mode"] == "semi_continuous":
        cont_symm_axis_idx = symm_params["continuous_symm_axis"]

        has_no_discrete = np.all(np.isclose(np.eye(3, dtype=np.float64)[None, :], symm_reductions))
        if not has_no_discrete:

            cont_axes = poses[:, :3, cont_symm_axis_idx]
            ref_axis = ref_pose[:, cont_symm_axis_idx]

            first_axis_dist = [
                np.linalg.norm(cont_axes - ref_pose[:, cont_symm_axis_idx][None, :], axis=1),
                np.linalg.norm(ref_pose[:, cont_symm_axis_idx][None, :] + cont_axes, axis=1)
            ]
            second_axis_dist = [
                np.linalg.norm(cont_axes - ref_pose[:, (cont_symm_axis_idx + 1) % 3][None, :], axis=1),
                np.linalg.norm(ref_pose[:, (cont_symm_axis_idx + 1) % 3][None, :] + cont_axes, axis=1)
            ]

            to_flip_first = first_axis_dist[0] > first_axis_dist[1]
            tie_breaker = np.isclose(first_axis_dist[0], first_axis_dist[1])
            to_flip_second = second_axis_dist[0] < second_axis_dist[1]
            to_flip_first[tie_breaker] = to_flip_second[tie_breaker]
            cont_axes[to_flip_first] *= -1.0

            ref_axis_broadcasted = np.broadcast_to(ref_axis, cont_axes.shape)
            poses[:, :3, :3] = vectorized_rot_between(ref_axis_broadcasted, cont_axes) @ ref_pose[None, :]
        else:
            removal_axes = ["xyx", "yxy", "zyz"][cont_symm_axis_idx]

            eulers = R.from_matrix(poses[:, :3, :3]).as_euler(removal_axes)
            eulers[:, 0] = 0
            poses[:, :3, :3] = R.from_euler(removal_axes, eulers).as_matrix()

    else:
        all_poses = poses[:, None, :3, :3] @ symm_reductions[None, :]
        dists = np.sum((all_poses - ref_pose[None, :])**2, axis=(-2, -1))

        best_symm_reduce = symm_reductions[np.argmin(dists, axis=1)]
        poses[:, :3, :3] = poses[:, :3, :3] @ best_symm_reduce

    return poses.astype(np.float64)

def add(pose_est, pose_gt, pts):
    """Average Distance of Model Points for objects with no indistinguishable
    views - by Hinterstoisser et al. (ACCV'12).

    Note: Please remove any symmetries beforehand!

    :param pose_est: 4x4 ndarray with the estimated pose transform matrix.
    :param pose_gt: 4x4 ndarray with the ground-truth pose transform matrix.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    vertices = np.c_[pts, np.ones(len(pts))]
    pose_diff = pose_est - pose_gt
    return np.mean(np.linalg.norm(pose_diff@vertices.T, axis=0))

def mvd(pose_est, pose_gt, pts):
    """Maximum Vertex Distance.

    Note: Please remove any symmetries beforehand!

    :param pose_est: 4x4 ndarray with the estimated pose transform matrix.
    :param pose_gt: 4x4 ndarray with the ground-truth pose transform matrix.
    :param pts: nx3 ndarray with 3D model vertex points.
    :return: The calculated error.
    """
    vertices = np.c_[pts, np.ones(len(pts))]
    # poses = [pose_est, pose_gt]
    pose_diff = pose_est - pose_gt
    return np.max(np.linalg.norm(pose_diff@vertices.T, axis=0))

class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)