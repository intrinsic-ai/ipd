from __future__ import annotations

from .constants import IPDCamera, IPDImage, IPDLightCondition, CameraFramework, DATASET_IDS
from .utils import download_dataset, download_cads, extract, DisableLogger

import os

import json
import numpy as np
from functools import cached_property
# from PIL import Image
import cv2
import logging
from typing import Optional, Union
logger = logging.getLogger(__name__)


# def set_logging_format(level=logging.INFO):
#   importlib.reload(logging)
#   FORMAT = '[%(funcName)s()] %(message)s'
#   logging.basicConfig(level=level, format=FORMAT)

# set_logging_format()

class IPDataset:
    """This class provides access to the IPDataset, which contains images, depth, and ground truth information for various scenes and objects.

    The dataset can be downloaded using the `download` argument. If the dataset is not found in the specified root directory, it will be downloaded and extracted.

    The dataset has the following properties:

        - `scenes`: A dictionary of scene IDs and their corresponding paths.
        - `objects`: A list of tuples containing the object part names and object IDs.
        - `K`: The camera intrinsics matrix.

    The dataset can be used to get the following information:

        - `get_camera()`: Returns the camera intrinsics, distortion, and pose in the specified framework.
        - `get_scene_labels()`: Returns the labels for the specified scene ID.
        - `get_img_path()`: Returns the image path for the specified scene ID and image type.
        - `get_img()`: Returns the image for the specified scene ID and image type.
        - `get_depth()`: Returns the depth for the specified scene ID.
        - `render()`: Renders the specified scene with the specified object.
        - `render_masks()`: Renders masks for all objects in the specified scene.
        - `create_masks()`: Creates masks for all objects in all scenes in the dataset.
        - `get_mask_path()`: Returns the file path of the mask for the specified object in the specified scene.
        - `get_mask()`: Returns the mask for the specified object in the specified scene.
    """

    def __init__(
            self,
            root: Union[str, os.PathLike] = "datasets",
            dataset_id: str = "dataset_basket_0",
            camera: IPDCamera = IPDCamera.BASLER_LR1,
            lighting: IPDLightCondition = IPDLightCondition.ALL,
            resize: float = 1.0,
            download: bool = False
        ) -> None:
        """Initializes an IPDataset object.

        Args:
            root (Union[str, os.PathLike], optional): The root directory of the dataset. Defaults to "datasets".
            dataset_id (str, optional): The ID of the dataset to load. Defaults to "dataset_basket_0".
            camera (IPDCamera, optional): The camera to use. Defaults to IPDCamera.BASLER_LR1.
            lighting (IPDLightCondition, optional): The lighting condition to use. Defaults to IPDLightCondition.ALL.
            resize (float, optional): The resize factor for the images. Defaults to 1.0 (no resizing)
            download (bool, optional): Whether to download the dataset if it is not found. Defaults to False.

        Raises:
            FileNotFoundError: If the dataset is not found and `download` is False.
            AssertionError: If the dataset ID is invalid.
        """
        
        self.root = root
        assert dataset_id in DATASET_IDS, f"Invalid dataset id {dataset_id}, must be one of {DATASET_IDS}"
        self.dataset_id = dataset_id
        self.dataset_path = os.path.join(root, dataset_id, "test")
        self.camera = camera
        self.lighting = lighting
        self.resize = resize

        if not os.path.exists(os.path.join(self.scenes[next(iter(self.scenes))], self.camera.folder)): #TODO doesn't work without any existing folders.
            if not download:
                raise FileNotFoundError(f"Dataset {self.dataset_id} for camera {self.camera.type} not found at {self.scenes[next(iter(self.scenes))]}/{self.camera.folder}, please download it first.")
            else:
                zip_path = download_dataset(dataset_id, self.camera.type, root)
                if zip_path:
                    extract(zip_path, root)
                    os.remove(zip_path)
                download_cads(root)

        logger.info(
                    f"\n\tDataset Path:\t{self.dataset_path}"+
                    f"\n\tCamera Type:\t{self.camera.type} \t {self.camera.folder}"+
                    f"\n\tNum Scenes:\t{len(self.scenes)}"
                )
    
    def __len__(self) -> int:
        return len(self.scenes)
    
    @cached_property
    def scenes(self) -> dict[int, str]:
        """Returns a dictionary of scene IDs and their corresponding paths for the chosen lightint condition.

        Returns:
            dict[int, str]: A dictionary of scene IDs and their corresponding paths.
        
        Example:
            print(dataset.scenes)
            # Output:
            # {
            #   0:  './datasets/dataset_basket_0/test/000000',
            #   ...
            #   29: './datasets/dataset_basket_0/test/000029',
            # }
        """
        scenes_paths = sorted([f.path for f in os.scandir(self.dataset_path) if f.is_dir()], key=lambda x: int(os.path.basename(x)))
        scene_ids = [int(os.path.basename(p)) for p in scenes_paths]
        scenes = {scene_id: scene_path for scene_id, scene_path in zip(scene_ids, scenes_paths) if scene_id in self.lighting.scenes}
        return scenes

    @cached_property
    def objects(self) -> list[tuple[str, int]]:
        """ Returns an ordered list of tuples containing the object part names and object ids (to identify between multiple instances for a given a part).
        
        Note:
            Used to index through o2c_by_object returned in IPDataset.get_scene_labels(...) 
        
        Returns:
            list[tuple[str, int]]: An ordered list of tuples containing the object part names and object ids.
        
        Example:
            print(dataset.objects)
            # Output:
            # [ 
            #   ('corner_bracket', 0), 
            #   ('corner_bracket', 1), # 2nd instance of corner_bracket
            #   ('gear1', 0), 
            #   ('gear2', 0) 
            # ]
        """
        # open the dataset_info.json
        info_path = os.path.join(os.path.dirname(self.dataset_path), "dataset_info.json")
        with open(info_path) as f:
            info = json.load(f)
        objs = [(part["part_id"], object_id) for part in info for object_id in range(int(part["part_count"])) ]
        return objs
    
    @cached_property
    def K(self) -> np.ndarray:
        K, _, _ = self.get_camera()
        return K

    def get_mesh_file(self, object_name:str) -> os.PathLike:
        """ Returns the mesh file for the specified object.
        
        Args:
            object_name (str): The name of the part to get the mesh file for.
        
        Returns:
            os.PathLike: The path to the mesh file.
        """
        mesh_file = os.path.join(self.root, "models", f"{object_name}.stl")
        if os.path.exists(mesh_file):
            logger.debug(f"Found mesh file for {object_name}")
            return mesh_file
        logger.warn(f"Mesh file for {object_name} not found at {mesh_file}")
        return None

    def get_camera(self, framework: CameraFramework = CameraFramework.OPENCV) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns camera intrinsics, distortion, and pose in the specified framework.

        Args:
            framework (CameraFramework, optional): 
                The camera framework to return the camera info in. Defaults to CameraFramework.OPENCV.
                See ipd.constants.CameraFramework for available frameworks.

        Returns:
            K (np.ndarray(3,3)):
                Camera intrinsics 
            d (np.ndarray(?)):
                Camera distortion
            c2w (np.ndarray(4, 4)):
                Pose of camera relative to world frame, converted to given camera framework. 
            
        """
        output = {}
        first_scene_id = next(iter(self.scenes))
        first_scene_path = self.scenes[first_scene_id]
        camera_path = os.path.join(first_scene_path, self.camera.folder)
        camera_json = os.path.join(camera_path, "scene_camera.json")
        with open(camera_json) as f:
            camera_params = json.load(f)['0']
        intrinsics = camera_params['cam_K']
        K = np.array(intrinsics).reshape((3, 3)) 
        dist = np.array(camera_params["cam_dist"])
        
        r = np.array(camera_params['cam_R_c2w']).reshape((3, 3))
        t = np.array(camera_params['cam_t_c2w']) 
        c2w = np.eye(4)  
        c2w[:3, :3] = r
        c2w[:3, 3] = t

        c2w = CameraFramework.convert(c2w, CameraFramework.OPENCV, framework)
        return K, dist, c2w
   
    def get_scene_labels(self, scene_id: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Returns the labels for the specified scene ID. 
        
        Note:
            In the paper, the `scene` is interchangeable with the `gripper` so c2s == c2g.

        Args:
            scene_id (int): The scene ID to get the information for.

        Returns:
            c2s (np.ndarray(4,4)):
                Camera pose to the scene/gripper frame
            o2c_by_object ( dict[ (str, int), np.ndarray(4,4)] ):
                A dictionary mapping object (part names, object IDs) tuples to their corresponding poses relative to the camera.
                    The keys are tuples opf object part names (e.g., "corner_bracket", "gear1") and object IDs (integers) within that part.
                    The values are 4x4 homogeneous transformation matrices representing the pose of each object relative to the camera.

        """
        scene_path = self.scenes[scene_id]
        scene_json = os.path.join(scene_path, str(self.camera.folder), "scene_pose.json")
        with open(scene_json) as f:
            scene = json.load(f)['0']
        r = np.array(scene['rob_R_r2s']).reshape((3, 3))
        t = np.array(scene['rob_T_r2s']) 
        r2s = np.eye(4)
        r2s[:3, :3] = r
        r2s[:3, 3] = t
        
        r = np.array(scene['cam_R_c2r']).reshape((3, 3))
        t = np.array(scene['cam_T_c2r']) 
        c2r = np.eye(4)
        c2r[:3, :3] = r
        c2r[:3, 3] = t

        c2s = r2s @ c2r #T_sc = T_sr @ T_rc 
        
        gt_json = os.path.join(scene_path, str(self.camera.folder), "scene_gt.json")
        o2c_by_object = {}
        with open(gt_json) as f:
            objects = json.load(f)['0']
        for obj in objects:
            r = np.array(obj['cam_R_m2w']).reshape((3, 3))
            t = np.array(obj['cam_t_m2w']) 
            o2c = np.eye(4)
            o2c[:3, :3] = r
            o2c[:3, 3] = t
            o2c_by_object[(obj["obj_name"], int(obj["obj_id"]))] = o2c

        return c2s, o2c_by_object

    def get_img_path(self, scene_id:int, image_type: Optional[IPDImage] = None) ->  Union[str, os.PathLike] : 
        """ Returns the image path for the specified scene ID and image type.

        Args:
            scene_id (int): The scene ID to get the image for.
            image_type (IPDImage, optional): 
                The type of image to return. Defaults to None, which will return the first valid image for the current camera.
                See ipd.constants.IPDImage for available image types. 

        Raises:
            AssertionError: If the image type is invalid for the camera. Valid image types are listed in self.camera.images.
                See ipd.constants.IPDCamera.

        Returns:
            Image: The path to the image for the specified scene ID and image type.
        
        Usage:
            from PIL import Image
            dataset = IPDataset(dataset_id = "dataset_basket_0", camera = IPDCamera.BASLER_LR1)
            img_path = dataset.get_img_path(scene_id=0, image_type=IPDImage.EXPOSURE_80)
            img = Image.open(img_path)
        """
        assert image_type is None or image_type in self.camera.images, f"Invalid image type {image_type} for camera {self.camera.name}, must be one of {self.camera.images}"
        if image_type is None:
            image_type = self.camera.images[0]
            logger.info(f"No image type specified, using {image_type} for camera {self.camera.name}")
        scene_path = self.scenes[scene_id]
        img_path = os.path.join(scene_path, str(self.camera.folder), image_type.filename)
        return img_path

    def get_img(self, scene_id:int, image_type: Optional[IPDImage] = None, resize:float=None, convert_hdr:bool=True) -> np.ndarray[float] :
        """ Returns the image for the specified scene ID and image type.
        
        Args:
            scene_id (int): The scene ID to get the image for.
            image_type (IPDImage, optional): 
                The type of image to return. Defaults to None, which will return the first image for the camera.
                See ipd.constants.IPDImage for available image types. 
            resize (float): 
                The resize factor for the image. Defaults to 1 (no resizing). 
            convert_hdr (bool): 
                Whether to convert HDR images to LDR. Defaults to True.
                If False, HDR images will be returned as is.

        Note:
            HDR images are returned as float32 arrays with values in the range [0, 1].
            LDR images are returned as uint8 arrays with values in the range [0, 255].


        Raises:
            AssertionError: If the image type is invalid for the camera. Valid image types are listed in self.camera.images.
                See ipd.constants.IPDCamera.

        Returns:
            img (np.ndarray[uint8 or float32]): The image for the specified scene ID and image type.
        """
        if resize is None:
            resize = self.resize
        
        img_path = self.get_img_path(scene_id, image_type)
        logger.info(f"Opening image from {img_path}")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if len(img.shape)==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if np.max(img) > 255 and convert_hdr:
            # Convert HDR to LDR
            gamma = 2.2
            tonemap = cv2.createTonemap(gamma)
            img = tonemap.process(img.astype(np.float32))
            img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        
        if resize != 1:
            img = cv2.resize(img, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST, dsize=None)

        return img
        
    def get_depth(self, scene_id:int, resize:float=None, znear:float=0.1, zfar:float=np.inf):
        """ Returns the depth for the specified scene ID.

        Args:
            scene_id (int): The scene ID to get the depth for.
            resize (float, optional): 
                The resize factor for the image. Defaults to self.resize (1 = no resizing).
            znear (float, optional):
                The near clipping plane for the depth. Defaults to 0.1.
            zfar (float, optional):
                The far clipping plane for the depth. Defaults to np.inf.
        Note:
            Returns None and logs an error if the camera does not have a depth file.

        Returns:
            depth (np.ndarray[float32]): The depth for the specified scene ID.
        """
        if self.camera != IPDCamera.PHOTONEO:
            logger.error(f"No depth file available for {self.camera}", exc_info=True)
            return None
        
        if resize is None:
            resize = self.resize

        depth_path = self.get_img_path(scene_id, image_type=IPDImage.PHOTONEO_DEPTH)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if resize!=1:
            depth = cv2.resize(depth, fx=resize, fy=resize, dsize=None, interpolation=cv2.INTER_NEAREST)
        depth[depth<znear] = 0.0
        depth[depth>zfar] = 0.0
        return depth

    def render(self, 
               scene_id:int, 
               obj_name:str = None,
               obj_id:int = None
        ) -> tuple[np.ndarray, np.ndarray]:
        """ Renders the specified scene with the specified object.

        Note:
            This method requires the following libraries to be installed:
                - pyrender
                - trimesh

            You can install these libraries using:
                `pip install ipd[render]` or `pip install pyrender trimesh`
                (or from source with `pip install -e .[render]`)

        Args:
            scene_id (int): The scene ID to render.
            obj_name (str, optional): The name of the object to render. Defaults to None, which will render all objects.
            obj_id (int, optional): The ID of the object to render. Defaults to None, which will render all objects.

        Returns:
            color (h, w, 3) uint8 or (h, w, 4) uint8: 
                The color buffer in RGB format, 
                or in RGBA format if :attr:`.RenderFlags.RGBA` is set.
                Not returned if flags includes :attr:`.RenderFlags.DEPTH_ONLY`.
            depth (h, w) float32: 
                The depth buffer in linear units.
        
        Usage:
            import matplotlib.pyplot as plt
            dataset = IPDataset(dataset_id = "dataset_basket_0", camera = IPDCamera.BASLER_LR1)
            color, depth = dataset.render(scene_id=0, obj_name="corner_bracket", obj_id=0)
            im = dataset.get_img(scene_id=0, image_type=IPDImage.EXPOSURE_80)
            fig = plt.figure(figsize=(10, 5))
            plt.axis('off')
            plt.imshow(im, alpha=.7)
            plt.imshow(color, alpha=1)

        
        """
        # check that rendering libraries are installed, if not, prompt user to install optional dependencies
        try:
            if os.environ['PYOPENGL_PLATFORM'] not in ["egl", "osmesa"]:
                logger.warn("If you want to use OSMesa or EGL, you need to set the PYOPENGL_PLATFORM environment variable before importing pyrender or any other OpenGL library. See https://pyrender.readthedocs.io/en/latest/examples/offscreen.html")
            import pyrender, trimesh
        except ImportError:
            print("The required libraries 'pyrender' and 'trimesh' are not installed. Please install them first using:")
            print("`pip install ipd[render]` or `pip install pyrender trimesh`")
            return None, None

        with DisableLogger():
            img = self.get_img(scene_id, self.camera.images[0])
        
        height, width = img.shape[:2]
        r = pyrender.OffscreenRenderer(width, height)
        _, o2c_by_object = self.get_scene_labels(scene_id)
        
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0])

        K, _, _ = self.get_camera()
        logger.debug(f"K: {K}")
        c2w = np.eye(4) 
        c2w = CameraFramework.convert(c2w, CameraFramework.OPENCV, CameraFramework.OPENGL)
        icamera = pyrender.IntrinsicsCamera(fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2], zfar=10000)
        scene.add(icamera, pose=c2w)

        for part, id in o2c_by_object:
            if obj_name and part != obj_name: # check if matches filter
                continue
            
            mesh_file = self.get_mesh_file(part)
            if mesh_file is None:
                continue
            part_trimesh = trimesh.load(mesh_file)
            mesh = pyrender.Mesh.from_trimesh(part_trimesh)

            if obj_id and id != obj_id:
                continue
        
            o2c = o2c_by_object[(part, id)]
            scene.add(mesh, pose=o2c)
                
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.RGBA)

        return np.array(color), np.array(depth)
        
    def render_masks(self, 
               scene_id:int
        ) ->  dict[ tuple[str, int] , np.ndarray ]:
        """ Renders masks for all objects in the specified scene.

        Note:
            This method requires the following libraries to be installed:
                - pyrender
                - trimesh

            You can install these libraries using:
                `pip install ipd[render]` or `pip install pyrender trimesh`
                (or from source with `pip install -e .[render]`)

        Args:
            scene_id (int): The scene ID to render masks for.

        Returns:
            masks_by_object ( dict[ (str, int), np.ndarray[h, w, 3] ] ):
                Dictionary of masks for each object indexed by (part name, object ID)

        Usage:
            import matplotlib.pyplot as plt
            dataset = IPDataset(dataset_id = "dataset_basket_0", camera = IPDCamera.BASLER_LR1)
            masks = dataset.masks(scene_id=0)
            mask = masks[dataset.objects[0]]  # get first mask
            fig = plt.figure(figsize=(10, 5))
            plt.axis('off')
            plt.imshow(mask)

        """
        # check that rendering libraries are installed, if not, prompt user to install optional dependencies
        try:
            if os.environ['PYOPENGL_PLATFORM'] not in ["egl", "osmesa"]:
                logger.warn("If you want to use OSMesa or EGL, you need to set the PYOPENGL_PLATFORM environment variable before importing pyrender or any other OpenGL library. See https://pyrender.readthedocs.io/en/latest/examples/offscreen.html")
            import pyrender, trimesh
        except ImportError:
            print("The required libraries 'pyrender' and 'trimesh' are not installed. Please install them first using:")
            print("`pip install ipd[render]` or `pip install pyrender trimesh`")
            return None, None

        with DisableLogger():
            img = self.get_img(scene_id, self.camera.images[0])
        
        height, width = img.shape[:2]
        r = pyrender.OffscreenRenderer(width, height)
        _, o2c_by_object = self.get_scene_labels(scene_id)
        
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0])

        K, _, _ = self.get_camera()
        logger.debug(f"K: {K}")
        c2w = np.eye(4) 
        c2w = CameraFramework.convert(c2w, CameraFramework.OPENCV, CameraFramework.OPENGL)
        icamera = pyrender.IntrinsicsCamera(fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2], zfar=10000)
        scene.add(icamera, pose=c2w)

        node_dict = {}
        for part, id in o2c_by_object:
            mesh_file = self.get_mesh_file(part)
            if mesh_file is None:
                continue
            part_trimesh = trimesh.load(mesh_file)
            mesh = pyrender.Mesh.from_trimesh(part_trimesh)
        
            o2c = o2c_by_object[(part, id)]
            node = pyrender.Node(mesh=mesh, matrix=o2c)

            node_dict[(part, id)] = node
            scene.add_node(node)
                

        seg_masks_by_object = {}
        for part, id in o2c_by_object:
            key = (part, id)
            node = node_dict[key]
            nodes = {node: 255}
            seg_mask = r.render(scene, pyrender.constants.RenderFlags.SEG, nodes)
            seg_masks_by_object[key] = seg_mask[0]
        
        
        return seg_masks_by_object
        
    def saves_masks(self, overwrite:bool=False) -> None:
        """Creates and saves masks for all objects in all scenes in the dataset

        Note:
            Masks are saved as PNG images to the following path:
                [self.root]/[self.dataset_id]/test/[scene]/[self.camera.folder]/mask/[obj_name]/[obj_id].png
            For example:
                ./datasets/dataset_basket_0/test/0/Basler-LR/mask/corner_bracket/0.png

        Args:
            overwrite (bool, optional): Whether or not to overwrite saved masks. Defaults to False.
        """
        for scene_id in self.scenes:
            masks = self.render_masks(scene_id)

            for obj_name, obj_id in masks:
                
                part_path = os.path.join(self.scenes[scene_id], str(self.camera.folder), "mask", obj_name)
                file_path = os.path.join(part_path, f"{obj_id}.png")
                if os.path.exists(file_path) and not overwrite:
                    logger.info(f"Mask already exists at {file_path}, skipping")
                    continue
                elif os.path.exists(file_path) and overwrite:
                    logger.warn(f"Mask already exists at {file_path}, overriding")

                os.makedirs(part_path, exist_ok=True)
                im = masks[(obj_name, obj_id)]
                # im = Image.fromarray(masks[(obj_name, obj_id)])
                # im.save(file_path)
                cv2.imwrite(file_path, im)

                logger.info(f"Mask saved to {file_path}")
    
    def get_mask_path(self, scene_id:int, object_name:str, object_id:int) -> os.PathLike:
        """Returns the file path of the mask for the specified object in the specified scene.

        Args:
            scene_id (int): The scene ID to get the mask for.
            object_name (str): The part name of the object to get the mask for.
            object_id (int): The object ID to get the mask for.

        Returns:
            mask_file (os.PathLike): The path to the mask for the specified object in the specified scene.
        """
        part_path = os.path.join(self.scenes[scene_id], str(self.camera.folder), "mask", object_name)
        mask_path = os.path.join(part_path, f"{object_id}.png")
        if not os.path.exists(mask_path):
            logger.error(f"No mask found for {object_name} {object_id} have you already created masks? (ipd.IPDataset.create_masks): {mask_path}", exc_info=True)
            return None
        return mask_path

    def get_mask(self, scene_id:int, object_name:str, object_id:int, resize:float=None, detect_bounding_box:bool=False) -> np.ndarray:
        """Returns the mask for the specified object in the specified scene.

        Args:
            scene_id (int): The scene ID to get the mask for.
            object_name (str): The part name of the object to get the mask for.
            object_id (int): The object ID to get the mask for.
            resize (float, optional): 
                The resize factor for the image. Defaults to self.resize (1 = no resizing). 
            detect_box (bool, optional):
                Whether to return a bounding box mask instead of an object mask. Defaults to False.
                
                
        Returns:
            np.ndarray[bool]: The mask for the specified object in the specified scene.
        """
        if resize is None:
            resize = self.resize

        mask_file = self.get_mask_path(scene_id, object_name, object_id)
        if mask_file is None:
            return None

        mask = cv2.imread(mask_file, -1)
        if mask is None:
            return None
        
        if resize !=1:
            mask = cv2.resize(mask, fx=resize, fy=resize, dsize=None, interpolation=cv2.INTER_NEAREST)

        if len(mask.shape)>2:
            mask = mask[:,:,0]

        if detect_bounding_box:
            H,W = mask.shape[:2]
            vs,us = np.where(mask>0)
            umin = us.min()
            umax = us.max()
            vmin = vs.min()
            vmax = vs.max()
            valid = np.zeros((H,W), dtype=bool)
            valid[vmin:vmax,umin:umax] = 1
        else:
            valid = mask>0
        
        # old code
        # mask = np.array(Image.open(mask_path).convert('RGBA'))
        # # maybe convert rgb2bgr?  red = mask[:,:,0].copy(); mask[:,:,0] = mask[:,:,2].copy(); mask[:,:,2] = red;

        return valid
