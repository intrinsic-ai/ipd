from __future__ import annotations
from aenum import Enum
import numpy as np

DATASET_IDS = [
    "dataset_basket_0",
    "dataset_basket_1",
    "dataset_basket_2",
    "dataset_basket_3",
    "dataset_basket_4",
    "dataset_basket_5",
    "dataset_basket_6",
    "dataset_basket_7",
    "dataset_basket_8",
    "dataset_basket_9",
    "dataset_darkbg_0",
    "dataset_darkbg_1",
    "dataset_darkbg_2",
    "dataset_darkbg_3",
    "dataset_darkbg_4",
    "dataset_darkbg_5",
    "dataset_darkbg_6",
    "dataset_darkbg_7",
    "dataset_darkbg_8",
    "dataset_texturedbg_0",
    "dataset_texturedbg_1",
    "dataset_texturedbg_2",
    "dataset_texturedbg_3"
]

CAD_FILES = [
    "corner_bracket",
    "corner_bracket0",
    "corner_bracket1",
    "corner_bracket2",
    "corner_bracket3",
    "corner_bracket4",
    "corner_bracket6",
    "gear1",
    "gear2",
    "handrail_bracket",
    "hex_manifold",
    "l_bracket",
    "oblong_float",
    "pegboard_basket",
    "pipe_fitting_unthreaded",
    "single_pinch_clamp",
    "square_bracket",
    "t_bracket",
    "u_bolt",
    "wraparound_bracket",
]


class CameraFramework(Enum):
    """Camera Frameworks.

    OPENCV: OpenCV camera framework.
    COLMAP: COLMAP camera framework.
    PYTORCH3D: PyTorch3D camera framework.
    OPENGL: OpenGL camera framework.

    Properties:
        value: The value of the camera framework.
    
    Static Methods:
        convert: Converts the camera pose from one framework to another.
        _flip_R: Internal method to flip the rotation matrix based on the specified flags. Used by `CameraFramework.convert`.
        _flip_t: Internal method to flip the translation vector based on the specified flags. Used by `CameraFramework.convert`.
    
    """
    OPENCV = 1
    COLMAP = 1
    PYTORCH3D = 2
    OPENGL = 3

    @staticmethod
    def _flip_R(rotation_matrix: np.ndarray,
               flip_x: bool = False,
               flip_y: bool = False,
               flip_z: bool = False) -> np.ndarray:
        flipped_matrix = rotation_matrix.copy()
        if flip_x:
            flipped_matrix[1:3, :] = -flipped_matrix[1:3, :]
        if flip_y:
            flipped_matrix[[0, 2], :] = -flipped_matrix[[0, 2], :]
        if flip_z:
            flipped_matrix[:, [0, 1]] = -flipped_matrix[:, [0, 1]]
        return flipped_matrix
    
    @staticmethod
    def _flip_t(translation_vector: np.ndarray,
               flip_x: bool = False,
               flip_y: bool = False,
               flip_z: bool = False) -> np.ndarray:
        flipped_vector = translation_vector.copy()
        if flip_x:
            flipped_vector[0] = -flipped_vector[0]
        if flip_y:
            flipped_vector[1] = -flipped_vector[1]
        if flip_z:
            flipped_vector[2] = -flipped_vector[2]
        return flipped_vector
    
    @staticmethod
    def convert(T, from_camera: CameraFramework, to_camera: CameraFramework) -> np.ndarray:
        from_to = (from_camera, to_camera)
        transform = (False, False, False)
        if from_to == (CameraFramework.OPENCV, CameraFramework.PYTORCH3D) or\
        from_to == (CameraFramework.PYTORCH3D, CameraFramework.OPENCV):
            transform = (True, True, False)
        elif from_to == (CameraFramework.OPENCV, CameraFramework.OPENGL) or\
        from_to == (CameraFramework.OPENGL, CameraFramework.OPENCV):
            transform = (False, True, True)
        elif from_to == (CameraFramework.PYTORCH3D, CameraFramework.OPENGL) or\
        from_to == (CameraFramework.OPENGL, CameraFramework.PYTORCH3D):
            transform = (True, False, True)

        if transform == (False, False, False):
            return T
    
        r = CameraFramework._flip_R(T[:3, :3], flip_x=transform[0], flip_y=transform[1], flip_z=transform[2])
        t = CameraFramework._flip_t(T[:3, 3], flip_x=transform[0], flip_y=transform[1], flip_z=transform[2])
        output = np.eye(4)
        output[:3, :3] = r
        output[:3, 3] = t
        return output


class IPDLightCondition(Enum):
    """IPD Light Conditions.

    DAY: Daylight condition.
    ROOM: Room light condition.
    SPOT: Spotlight condition.
    ALL: All light conditions.

    Properties:
        value: The value of the light condition.
        scenes: A list of scene IDs corresponding to the light condition.

    """
    _init_ = 'value scenes'
    DAY = 1, list(range(0,30))
    ROOM = 2, list(range(30,60))
    SPOT = 3, list(range(60,90))
    ALL = 4, list(range(0,90))


class IPDImage(Enum):
    """IPD Image Types. 
    
    [EXPOSURE_1 ...EXPOSURE_200]    For FLIR and Basler cameras, images are captured using four exposures (1ms, 30ms, 80ms, 200ms). 
    [PHOTONEO_DEPTH, PHOTONEO_HDR]  For Photoneo, a depth map and 12-bit HDR image (for tone-mapping) is captured.

    Properties:
        filename: The filename of the image.
    """
    _init_ = 'value filename'
    EXPOSURE_1 =     1, "rgb/0_1000.png"
    EXPOSURE_30 =    2, "rgb/0_30000.png"
    EXPOSURE_80 =    3, "rgb/0_80000.png"
    EXPOSURE_200 =   4, "rgb/0_200000.png"
    PHOTONEO_DEPTH = 5, "depth/000000.png"
    PHOTONEO_HDR =   6, "rgb/000000.png"

class IPDCamera(Enum):
    """IPD Camera Types. 

    [FLIR1 ...FLIR4]            4 Mono-Polar FLIR Cameras at 5MP resolution with baselines from 50cm to 1m.
    [Basler_HR1 ...Basler_HR5]  5 Basler RGB Cameras at 8MP with baselines from 10cm to 1m. 
    [Basler_LR1 ...Basler_LR3]  3 Basler RGB Cameras at 2MP with baselines from 10cm to 1m.
    [PHOTONEO]                  1 Photoneo XL giving 500um accurate depth at a 2m distance. [Photoneo]

    Properties:
        folder (str): The folder name corresponding to the camera.
        type (str): The type of the camera.
        images (list[IPDImage]): A list of IPDImage types that the camera captures.
    """
    _init_ = 'value folder type images'
    FLIR1 =      1,  '21095966', 'FLIR_Polar', [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]
    FLIR2 =      2,  '21192436', 'FLIR_Polar', [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]
    FLIR3 =      3,  '21192442', 'FLIR_Polar', [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]
    FLIR4 =      4,  '21196067', 'FLIR_Polar', [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]
    PHOTONEO =   5,  '000',      'Photoneo',   [IPDImage.PHOTONEO_HDR, IPDImage.PHOTONEO_DEPTH]
    BASLER_HR1 = 6,  '40406684', 'Basler-HR',  [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]
    BASLER_HR2 = 7,  '40406686', 'Basler-HR',  [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]
    BASLER_HR3 = 8,  '40406687', 'Basler-HR',  [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]
    BASLER_HR4 = 9,  '40406688', 'Basler-HR',  [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]
    BASLER_HR5 = 10, '40406689', 'Basler-HR',  [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]
    BASLER_LR1 = 11, '24466147', 'Basler-LR',  [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]
    BASLER_LR2 = 12, '24466154', 'Basler-LR',  [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]
    BASLER_LR3 = 13, '24466161', 'Basler-LR',  [IPDImage.EXPOSURE_1, IPDImage.EXPOSURE_30, IPDImage.EXPOSURE_80, IPDImage.EXPOSURE_200]

CAMERA_NAMES = ["FLIR_polar", "Photoneo", "Basler-HR", "Basler-LR"]