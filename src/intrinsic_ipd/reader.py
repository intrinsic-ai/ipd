from __future__ import annotations

from functools import lru_cache
import os, logging
import numpy as np
import pandas as pd
import xarray as xr
import json
import yaml
import itertools, operator
import cv2
from importlib import reload
import trimesh
from tqdm import tqdm


from typing import Union, Optional

from .constants import IPDCamera, IPDImage, IPDLightCondition, CameraFramework, DATASET_IDS
from .utils import download_dataset, download_cads, extract, DisableLogger, extract_symmetry_params, vectorized_remove_symmetry



class IPDReader:
    """
    This class provides access to the IPDataset, which contains images, depth, and ground truth information for various scenes and objects.

    The dataset can be downloaded using the `download` argument. If the dataset is not found in the specified root directory, it will be downloaded and extracted.

    The dataset reader class has the following properties:
        [Dataset Properties]
        - root (str): The root directory where the dataset is stored.
        - dataset_id (str): one of intrinsic_ipd.constants.DATASET_IDS
        - lighting (IPDLightCondition): one of intrinsic_ipd.constants.IPDLightCondition
        - scenes (dict[int, str]): mapping scene ids to scene paths
            i.e. {0: './datasets/dataset_basket_0/test/000000', ...}
        - objects (Iterable[tuple[str, int]]): A list of tuples containing the object part name and instance
            i.e. [("gear2", 0), ("gear2", 1), ... ("hex_manifold", 0)]
        - parts (Iterable[str]): All the parts in the dataset.
            i.e. ["gear2", "hex_manifold"]

        [Camera Properties]
        - camera (IPDCamera): one of intrinsic_ipd.constants.IPDCamera
        - cam_K (np.ndarray(3,3)): The camera intrinsics matrix.
        - cam_d (np.ndarray): The camera distortions.
        - cam_c2w (np.ndarray(4, 4)): The camera pose matrix.
        
        [Pose DataArrays]
        - o2c (xr.DataArray): object to camera poses for every scene and object 
            dims = ["scene", "object" = pd.MultiIndex["part", "instance"], "transform_major", "transform_minor"]
            shape = [#scenes, #objects, 4, 4]
        - c2g (xr.DataArray): camera to gripper pose for every scene.
            dims = ["scene", "transform_major", "transform_minor"]
            shape = [#scenes, 4, 4]
        - g2r (xr.DataArray): gripper to robot pose for every scene.
            dims = ["scene", "transform_major", "transform_minor"]
            shape = [#scenes, 4, 4]
        - o2g (xr.DataArray): object to gripper pose for every object.
            dims = ["object" = pd.MultiIndex["part", "instance"], "transform_major", "transform_minor"]
            shape = [#objects, 4, 4]
        - r2c (xr.DataArray): robot to camera pose.
            dims = ["transform_major", "transform_minor"]
            shape = [4, 4]
    
    This dataset reader class has the following methods:
        - get_img(scene): Returns the image for the specified scene ID and image type.
        - get_depth(scene): Returns the depth for the specified scene ID and depth type.
        - get_mask(scene, part, instance): Returns the mask for the specified object in the specified scene.
        - get_mesh(part): Returns the mesh for the specified object.
        - remove_symmetry(part, poses): Returns pose array or list with symmetry removed.
        - remove_symmetry_xarray(part, poses_xarr): Returns xarray with symmetry removed.
        - render_masks(scene): Renders masks for all objects in a scene.


    """

    def __init__(
            self,
            root: Union[str, os.PathLike] = "datasets",
            dataset_id: str = "dataset_basket_0",
            camera: IPDCamera = IPDCamera.BASLER_LR1,
            lighting: IPDLightCondition = IPDLightCondition.ALL,
            download: bool = False
        ) -> None:

        assert dataset_id in DATASET_IDS, f"Invalid dataset id {dataset_id}, must be one of {DATASET_IDS}"
        
        self.root = root
        self.dataset_id = dataset_id
        self.dataset_path = os.path.join(root, dataset_id, "test")
        self.camera = camera
        self.lighting = lighting
        
        self._check_and_download(download)
        self.scenes = self._load_scenes(self.lighting)
        self.objects = self._load_objects()
        self.parts = set([o[0] for o in self.objects])

        self.cam_K, self.cam_d, self.cam_c2w = self._load_camera() 
        self.o2c = self._load_o2c()                                 #dims: [scene, object, transform_major, transform_minor]
        self.c2g, self.g2r, self.r2c = self._load_c2g_g2r_r2c()     #dims: [scene, transform_major, transform_minor]

        self.o2g = xr.apply_ufunc(np.matmul, self.c2g.isel(scene=0), self.o2c.isel(scene=0), 
               input_core_dims=[["transform_major", "transform_minor"],["object","transform_major", "transform_minor"]],
               output_core_dims=[["object", "transform_major", "transform_minor"]])

        self._check_and_render_masks(overwrite=False)
        self.__name__ = '/'.join([dataset_id, camera.name, lighting.name])

        logging.info(
                    f"\n\tDataset:\t{self.dataset_id}"
                    f"\n\tDataset Path:\t{self.dataset_path}"+
                    f"\n\tCamera Type:\t{self.camera.type} (ID: {self.camera.folder})"+
                    f"\n\tLighting:\t{self.lighting.name}"+
                    f"\n\tNum Scenes:\t{len(self.scenes)}"
                )

    def __len__(self) -> int:
        return len(self.scenes)
    
    def assert_valid_scene(self, scene:int):
        assert scene in self.scenes, f"Scene {scene} not in dataset, try one of {self.scenes}"
    
    def assert_valid_part(self, part:str):
        assert part in self.parts, f"Part `{part}` not in dataset, try one of {self.parts}"
    
    def assert_valid_object(self, object:tuple[str, int]):
        self.assert_valid_part(object[0])
        objects_of_part = [o for o in self.objects if o[0] == object[0]]
        assert object in self.objects, f"Object {object} not in dataset, try one of {objects_of_part}"

    def _check_and_download(self, download: bool = False) -> None:
        """Checks if dataset is downloaded. If not, downloads it or raises an error.

        Args:
            download (bool, optional): Whether to download the dataset if it is not found. Defaults to False.

        Raises:
            FileNotFoundError: If no dataset is found and `download` is False.
        """
        check_exists_path = os.path.join(self.dataset_path, str(self.lighting.scenes[0]).zfill(6), self.camera.folder)
        if (not os.path.exists(check_exists_path)):
            if not download:
                raise FileNotFoundError(f"Dataset {self.dataset_id} for camera {self.camera.type} not found at {self.scenes[next(iter(self.scenes))]}/{self.camera.folder}, please download it first.")
            else:
                zip_path = download_dataset(self.dataset_id, self.camera.type, self.root)
                if zip_path:
                    extract(zip_path, self.root)
                    os.remove(zip_path)
                download_cads(self.root)

    def _check_and_render_masks(self, overwrite: bool = False) -> None:
        """ Create and save object level masks for every object and every scene. If overwrite is False, check if masks already exists and skip.
        Masks are saved to [scene]/[camera]/mask/[part]/[instance].png

        Args:
            overwrite (bool, optional): If existing mask files should be overwritten. Defaults to False.
        """
        to_render = []
        for scene in self.scenes:
            if not overwrite:
                for part, instance in self.objects:
                    part_path = os.path.join(self.scenes[scene], str(self.camera.folder), "mask", part)
                    file_path = os.path.join(part_path, f"{instance}.png")
                    os.makedirs(part_path, exist_ok=True)
                    if not os.path.exists(file_path):
                        try:
                            self.get_mesh(part)
                        except:
                            continue
                        to_render.append((scene, part, instance))
                    elif overwrite:
                        logging.warn(f"Mask already exists at {file_path}, overwriting")
                    else:
                        logging.debug(f"Mask already exists at {file_path}, skipping")
            else:
                for part, instance in self.objects:
                    try:
                        self.get_mesh(part)
                    except:
                        continue
                    to_render.append((scene, part, instance))
        if len(to_render) > 0:
            logging.info(f"Rendering {len(to_render)} masks...")
            try:
                with tqdm(total=len(to_render)) as pbar:
                    for scene, group in itertools.groupby(to_render, operator.itemgetter(0)):
                        masks = self.render_masks(scene)
                        for _, part, instance in group:
                            file_path = os.path.join(self.scenes[scene], str(self.camera.folder), "mask", part, f"{instance}.png")
                            if (part, instance) in masks:
                                im = masks[(part, instance)]
                                cv2.imwrite(file_path, im)
                                logging.debug(f"Mask saved to {file_path}")
                            pbar.update(1)
            except Exception as error:
                logging.error("Skipping mask rendering.", exc_info=error)
        
    def _load_scenes(self, lighting: IPDLightCondition = IPDLightCondition.ALL) -> dict[int, str]: 
        """Returns a dictionary of sorted scene IDs and their corresponding paths for the given lighting condition.
        
        Args:
            lighting (IPDLightCondition, optional): The lighting condition to filter on. Defaults to IPDLightCondition.ALL.

        Returns:
            dict[int, str]: A dictionary of scene IDs and their corresponding directory paths.
        
        Example:
            print(dataset.scenes)
            # Output:
            # {
            #   0:  './datasets/dataset_basket_0/test/000000',
            #   ...
            #   29: './datasets/dataset_basket_0/test/000029',
            # }
        """
        subdirs = [f.path for f in os.scandir(self.dataset_path) if f.is_dir()]
        scene_dirs = sorted(subdirs, key=lambda x: int(os.path.basename(x)))
        scenes = [int(os.path.basename(p)) for p in scene_dirs]
        # filter if in lighting condition
        scenes = {scene: scene_path for scene, scene_path in zip(scenes, scene_dirs) if scene in lighting.scenes}
        return scenes
    
    def _load_objects(self) -> tuple[list[str], list[tuple[str, int]]]:
        """ Returns an ordered list of tuples for the objects in the dataset.
        Each object is identified by (part:str, instance_id:int) to identify between multiple instances for a given a part.
    
        Note: 
            Can use itertools.groupby(self.objects, operator.itemgetter(0)) to get objects grouped by part.
        
        Returns:
            objects (list[tuple[str, int]]): _description_
        """
        # open dataset_info.json
        info_path = os.path.join(os.path.dirname(self.dataset_path), "dataset_info.json")
        with open(info_path) as f:
            info = json.load(f)
        objects = sorted([(part["part_id"], instance) for part in info for instance in range(int(part["part_count"]))])
        
        # check against the first scene, ground truth
        scene_path = next(iter(self.scenes.values()))
        camera = os.path.join(scene_path, str(self.camera.folder))
        gt_path = os.path.join(camera, "scene_gt.json")
        with open(gt_path) as f:
            objects_gt = sorted([(obj["obj_name"], int(obj["obj_id"])) for obj in json.load(f)['0']])
        if not objects_gt == objects:
            message = ""
            for part, group in itertools.groupby(objects, operator.itemgetter(0)):
                message += f"\n\t {part}: {len(list(group))}"
            logging.critical(f"\n{self.dataset_id}: dataset_info.json is incorrect, should be: {message}")
            return objects_gt
        return objects

    def _load_camera(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns camera intrinsics, distortion, and pose in the specified framework.

        Returns:
            K (np.ndarray(3,3)):
                Camera intrinsics 
            d (np.ndarray):
                Camera distortion
            c2w (np.ndarray(4, 4)):
                Pose of camera relative to world frame, converted to given camera framework. 
            
        """
        
        first_scene_path = next(iter(self.scenes.values()))
        camera_path = os.path.join(first_scene_path, self.camera.folder)
        camera_json = os.path.join(camera_path, "scene_camera.json")
        with open(camera_json) as f:
            camera_params = json.load(f)['0']
        intrinsics = camera_params['cam_K']
        K = np.array(intrinsics).reshape((3, 3)) 

        dist = np.array(camera_params["cam_dist"])
        
        # r = np.array(camera_params['cam_R_c2w']).reshape((3, 3))     #TODO This is wrong?
        # t = np.array(camera_params['cam_t_c2w'])                     #TODO This is wrong?
        c2w = np.eye(4)  
        # c2w[:3, :3] = r                                              #TODO This is wrong?
        # c2w[:3, 3] = t                                               #TODO This is wrong?

        return K, dist, c2w
    
    def _load_o2c(self) -> xr.DataArray:
        """Returns all object to camera poses for all scenes and all objects as an 4D xarray. 

        Note:
            The dimensions of the xarray are [scene, object, transform_major, transform_minor], where the last two axes are the pose axes.
            The shape of the xarray is [#scenes, #objects, 4, 4]
                
            The "scene" dimension can be referenced by integers from self.scenes.keys()
            The "object" dimension can be referenced by (part:str, instance:int) tuples from self.objects
            Furthermore, the object dimension has sub-levels "part" and "instance" which can individually be referenced. 
        
        Usage:
            reader = IPDReader(...)
            reader.o2c.sel(scene=30) 
            reader.o2c.sel(part="gear", scene=30) 
            reader.o2c.sel(part="gear", instance=0, scene=30) 
            reader.o2c.sel(part="gear", instance=0) 
            reader.o2c.sel(part="gear") 

        Returns:
            xr.DataArray: Object to Camera transforms for all objects across all scenes.
        """
        data = []
        object_order = None
        for s in self.scenes:
            scene_path = self.scenes[s]
            gt_json = os.path.join(scene_path, str(self.camera.folder), "scene_gt.json")
            with open(gt_json) as f:
                objects = json.load(f)['0']
            # Load all poses in the scene into a dictionary, indexed by object (tuple[part, instance])
            o2c_by_object = {}
            for obj in objects:
                r = np.array(obj['cam_R_m2w']).reshape((3, 3))  #TODO: Fix the data naming
                t = np.array(obj['cam_t_m2w']) 
                o2c = np.eye(4)
                o2c[:3, :3] = r
                o2c[:3, 3] = t
                o2c_by_object[(obj["obj_name"], int(obj["obj_id"]))] = o2c
            # Sort by object 
            o2c_by_object = dict(sorted(o2c_by_object.items()))
            # Make sure that order of objects is preserved across scenes
            if object_order is None:
                object_order = list(o2c_by_object.keys())
            else:
                assert object_order == list(o2c_by_object.keys()), "Scenes in dataset must have same objects for pose annotations"
            # Add to data
            data.append(list(o2c_by_object.values()))
        # Construct DataArray
        xarr = xr.DataArray(
            data = np.array(data),
            dims = ["scene", "object", "transform_major", "transform_minor"], #labeling each dimension of the 4D data array
            coords={
                "scene": list(self.scenes.keys()),
                "object" : pd.MultiIndex.from_tuples(self.objects, names=["part", "instance"]), #pandas multi-index for object tuples
                "transform_major": [0, 1, 2, 3],
                "transform_minor": [0, 1, 2, 3],
            }
        )
        return xarr
    
    def _load_c2g_g2r_r2c(self) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Private method to load camera-to-gripper, gripper-to-robot, robot-to-camera poses from `scene_pose.json`
        
        Note:
             data arrays have: 
                dims = ["scene", "transform_major", "transform_minor"] 
                shape = [#scenes, 4, 4]
        
        Returns:
            c2g_xarr (xr.DataArray): Camera to Gripper transforms for all scenes
                dims = ["scene", "transform_major", "transform_minor"] 
                shape = [#scenes, 4, 4]
            g2r_xarr (xr.DataArray): Gripper to Robot transforms for all scenes
                dims = ["scene", "transform_major", "transform_minor"] 
                shape = [#scenes, 4, 4]
            r2c_xarr (xr.DataArray): Robot to Camera transform
                dims = ["transform_major", "transform_minor"] 
                shape = [4, 4]
        """
        c2g_data = []
        g2r_data = []
        r2c = None
        for s in self.scenes:
            scene_path = self.scenes[s]
            scene_json = os.path.join(scene_path, str(self.camera.folder), "scene_pose.json")
            with open(scene_json) as f:
                scene = json.load(f)['0']
            r = np.array(scene['rob_R_r2s']).reshape((3, 3)) #TODO: Fix the data naming
            t = np.array(scene['rob_T_r2s']) 
            g2r = np.eye(4)
            g2r[:3, :3] = r
            g2r[:3, 3] = t
            
            if r2c is None:
                r = np.array(scene['cam_R_c2r']).reshape((3, 3))  #TODO: Fix the data naming
                t = np.array(scene['cam_T_c2r']) 
                r2c = np.eye(4)
                r2c[:3, :3] = r
                r2c[:3, 3] = t

            c2g =  np.linalg.inv(g2r) @ np.linalg.inv(r2c)  #T_sc = T_sr @ T_rc 

            c2g_data.append(c2g)
            g2r_data.append(g2r)
            

        dims = ["scene", "transform_major", "transform_minor"]
        coords = {
            "scene": list(self.scenes.keys()),
            "transform_major": [0, 1, 2, 3],
            "transform_minor": [0, 1, 2, 3],
        }
        c2g_xarr = xr.DataArray(data = np.array(c2g_data), dims = dims, coords = coords)
        g2r_xarr = xr.DataArray(data = np.array(g2r_data), dims = dims, coords = coords)
        r2c_xarr = xr.DataArray(data = np.array(r2c), dims = dims[1:], 
                                coords = {"transform_major": [0, 1, 2, 3],
                                          "transform_minor": [0, 1, 2, 3]})
        
        return c2g_xarr, g2r_xarr, r2c_xarr
    
    def _get_img_file(self, scene:int, image_type: Optional[IPDImage] = None) ->  Union[str, os.PathLike] : 
        """ Returns the image path for the specified scene ID and image type.

        Args:
            scene (int): The scene ID to get the image for.
            image_type (IPDImage, optional): 
                The type of image to return. Defaults to None, which will return the first valid image for the current camera.
                See ipd.constants.IPDImage for available image types. 

        Raises:
            AssertionError: If the image type is invalid for the camera. Valid image types are listed in self.camera.images.
                See ipd.constants.IPDCamera.
            FileNotFoundError: If no image file found

        Returns:
            os.PathLike: The path to the image for the specified scene ID and image type.
        
        Usage:
            from PIL import Image
            dataset = IPDataset(dataset_id = "dataset_basket_0", camera = IPDCamera.BASLER_LR1)
            img_path = dataset.get_img_path(scene=0, image_type=IPDImage.EXPOSURE_80)
            img = Image.open(img_path)
        """
        assert image_type is None or image_type in self.camera.images, f"Invalid image type {image_type} for camera {self.camera.name}, must be one of {self.camera.images}"
        if image_type is None:
            image_type = self.camera.images[0]
            logging.info(f"No image type specified, using {image_type} for camera {self.camera.name}")
        scene_path = self.scenes[scene]
        img_path = os.path.join(scene_path, str(self.camera.folder), image_type.filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found; {img_path}")
        return img_path

    @lru_cache
    def _get_symm_params(self, part: str) -> Union[dict, None]:
        """Returns the symmetry parameters for the specified object, or None if no file found.

        Args:
            part (str): The name of the part to get the symmetry transformations for.

        Returns:
            dict (see utils.extract_symmetry_params)
        """
        
        symm_json = os.path.join(self.root, "models", "symmetries", f"{part}_symm.json")
        if os.path.exists(symm_json):
            logging.debug(f"Found symmetries json for {part}")
            with open(symm_json) as f:
                symmetries_raw = json.load(f)
            return extract_symmetry_params(**symmetries_raw)
        logging.debug(f"Symmetries json for {part} not found at {symm_json}")
        return None
    
    def get_img(self, scene:int, image_type: Optional[IPDImage] = None, convert_hdr:bool=True, return_path:bool=False) -> tuple[np.ndarray[float], os.PathLike]:
        """ Returns the image and image path for the specified scene and image type.
        
        Args:
            scene (int): The scene ID to get the image for.
            image_type (IPDImage, optional): 
                The type of image to return. Defaults to None, which will return the first image for the camera.
                See ipd.constants.IPDImage for available image types. 
            convert_hdr (bool): 
                Whether to convert HDR images to LDR. Defaults to True.
                If False, HDR images will be returned as is.
            return_path (bool): 
                Whether to return the path to the image. Defaults to False.

        Note:
            HDR images are returned as float32 arrays with values in the range [0, 1].
            LDR images are returned as uint8 arrays with values in the range [0, 255].


        Raises:
            AssertionError: If the image type is invalid for the camera. Valid image types are listed in self.camera.images.
                See ipd.constants.IPDCamera.

        Returns:
            img (np.ndarray[uint8 or float32]): The image for the specified scene ID and image type.
            im_path (os.PathLike): Path to the image file, if return_path is True.
        """
        self.assert_valid_scene(scene)
        
        img_path = self._get_img_file(scene, image_type)
        logging.info(f"Opening image from {img_path}")
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
        
        if return_path:
            return img, img_path
        else:
            return img
    
    def get_depth(self, scene:int, znear:float=0.1, zfar:float=np.inf, return_path:bool=False) -> tuple[np.ndarray[float], os.PathLike]:
        """ Returns the depth for the specified scene ID and depth file path.

        Args:
            scene (int): The scene ID to get the depth for.
            znear (float, optional):
                The near clipping plane for the depth. Defaults to 0.1.
            zfar (float, optional):
                The far clipping plane for the depth. Defaults to np.inf.
            return_path (bool, optional):
                Whether to return the path to the depth file. Defaults to False.

        Returns:
            depth (np.ndarray[float32]): The depth for the specified scene ID.
            depth_path (os.PathLike): The path to the depth file for the specified scene ID, if return_path is True.
        
        Raises:
            FileNotFoundError: if no depth file found
        """
        self.assert_valid_scene(scene)

        if self.camera != IPDCamera.PHOTONEO:
            raise FileNotFoundError(f"No depth file available for {self.camera}", exc_info=True)

        depth_path = self._get_img_file(scene, image_type=IPDImage.PHOTONEO_DEPTH)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth[depth<znear] = 0.0
        depth[depth>zfar] = 0.0
        if return_path:
            return depth, depth_path
        else:
            return depth
    
    def get_mask(self, scene:int, part:str, instance:int, detect_bounding_box:bool=False, return_path:bool=False) -> tuple[np.ndarray[bool], os.PathLike]:
        """Returns the mask for the specified object in the specified scene.

        Args:
            scene (int): The scene ID to get the mask for.
            part (str): The part name of the object to get the mask for.
            instance (int): The part instance to get the mask for.
            detect_box (bool, optional):
                Whether to return a bounding box mask instead of an object mask. Defaults to False.
            return_path (bool, optional):
                Whether to return the path to the mask file. Defaults to False.

        Returns:
            valid np.ndarray[bool]: The mask for the specified object in the specified scene.
            mask_file (os.PathLike): The path to the mask for the specified object in the specified scene, if return_path is True.

        Raises:
            FileNotFoundError: if no mask file found
        """
        self.assert_valid_scene(scene)
        self.assert_valid_object((part, instance))

        part_path = os.path.join(self.scenes[scene], str(self.camera.folder), "mask", part)
        mask_file = os.path.join(part_path, f"{instance}.png")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"No mask found for {part} {instance} have you already created masks? (ipd.IPDataset._check_and_render_masks): {mask_file}") 

        mask = cv2.imread(mask_file, -1)
        if mask is None:
            return None
        
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
        
        if return_path:
            return valid, mask_file
        else:
            return valid

    def get_mesh(self, part:str, return_path:bool=False) -> tuple[trimesh.Mesh, os.PathLike]:
        """ Returns the mesh file for the specified object.
        
        Args:
            part (str): The name of the part to get the mesh file for.
            return_path (bool): Whether to return the path to the mesh file. Defaults to False
        
        Returns:
            mesh (trimesh.Mesh): The object mesh.
            mesh_file (os.PathLike): The path to the mesh file, if return_path is True.
            
        
        Raises:
            FileNotFoundError: If no mesh file is found for the part.
    
        """
        self.assert_valid_part(part)
        mesh_file = os.path.join(self.root, "models", f"{part}.stl")
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"No mesh file found for {part} at {mesh_file}")
        if return_path:
            return trimesh.load(mesh_file), mesh_file
        #special case:
        if part == "t_bracket":
            mesh = trimesh.load(mesh_file)

        return trimesh.load(mesh_file)

    def get_match_dist_thresh_by_part(self) -> dict[str, float]:
        """ Return dictionary of part-wise floats representing maximum distance to match for given part.

        Returns:
            dict[str, float]: Dictionary of thresholds to use matching predictions for each part.
        """
        with open(os.path.join(self.root, "models", "config.yaml")) as stream:
            config = yaml.safe_load(stream)
        thresh_by_part = config['match_threshold']
        return thresh_by_part
    

    def remove_symmetry(self,
                        part:str, 
                        poses:Union[list[np.ndarray], np.ndarray, xr.DataArray]) -> Union[np.ndarray, list, xr.DataArray] :
        """For a given part, reduces all poses that are considered rotationally symmetric with each other to the same pose.
        
        Note:
            For given symmetry parameters along the X, Y, and Z axes, three cases for symmetry removal:
                Fully continuous symmetries (aka sphere) --- Reduce poses to the reference pose.
                Mixture of continuous and discrete symmetry --- Flip continuous axis to be as close as possible to the reference pose's corresponding axis. Then adjust the reference pose so that its corresponding axis aligns with the flipped or non-flipped continuous axis.
                *Only* discrete symmetries --- Apply symmetry transforms then pick the pose closest to the reference pose (by the Frobenius Norm). 

        Args:
            part (str): Name of part to get symmetry parameters for.
            poses (Union[list[np.ndarray], np.ndarray]): A list [[4,4] x N] or vector [...,4,4] of poses (for given part) to remove symmetry from.


        Returns:
            np.ndarray or list: A vector or list of 4x4 poses for the part with symmetry removed. Shape and datatype is preserved from input.
        """
        self.assert_valid_part(part)
        
        symm_params = self._get_symm_params(part)
        if symm_params is None:
            return poses
        
        if isinstance(poses, xr.DataArray):
            def remove_symmetry(array):
                logging.debug(f"received {type(array)} shape: {array.shape}")
                result = vectorized_remove_symmetry(array.reshape(-1, 4, 4), symm_params).reshape(array.shape)
                logging.debug(f"result.shape: {result.shape}")
                return result
            
            return xr.apply_ufunc(remove_symmetry, poses, 
                           input_core_dims=[["transform_major", "transform_minor"]], 
                           output_core_dims=[[ "transform_major", "transform_minor"]],   
                           )
        
        if isinstance(poses, list):
            reshape = "list"
            poses = np.array(poses)
    
        else: # is np.ndarray
            assert poses.shape[-1] == 4 and poses.shape[-2] == 4, "must be an array of shape (..., 4, 4)"
            if len(poses.shape) == 2:
                reshape = poses.shape
                poses = poses.expand_dims(0)
            elif len(poses.shape) > 3: 
                reshape = poses.shape
                poses = poses.reshape(-1, 4, 4)
            else:
                reshape = None

        poses = vectorized_remove_symmetry(poses, symm_params)

        if reshape == "list":
            poses = list(poses)
        elif reshape is not None:
            poses = poses.reshape(reshape)
        return poses
    
    def remove_symmetry_xarray(self,
                                poses_xarr:xr.DataArray) -> xr.DataArray:
        """For a pose xarray (such as self.o2c), reduces all poses that are considered rotationally symmetric with each other to the same pose.

        Args:
            poses_xarr (xr.DataArray): DataArray with dimensions [scene, object(part,instance), transform_major, transform_minor] of shape [#scenes, #objects, 4, 4]

        Returns:
            xr.DataArray: DataArray of 4x4 poses with symmetry removed per part. Dimensions and shape are preserved from input.
        """
        return poses_xarr.groupby("part").map(
            lambda group: self.remove_symmetry(group.part.data[0], group))
         
    def render_masks(self, 
               scene:int
        ) ->  dict[ tuple[str, int] , np.ndarray]:
        """ Renders masks for all objects in the specified scene.

        Args:
            scene (int): The scene ID to render masks for.
            
        Returns:
            masks_by_object ( dict[ (str, int), np.ndarray[h, w, 3] ] ):
                Dictionary of masks for each object indexed by (part name, object ID)

        Usage:
            import matplotlib.pyplot as plt
            dataset = IPDataset(dataset_id = "dataset_basket_0", camera = IPDCamera.BASLER_LR1)
            masks = dataset.masks(scene=0)
            mask = masks[dataset.objects[0]]  # get first mask
            fig = plt.figure(figsize=(10, 5))
            plt.axis('off')
            plt.imshow(mask)

        """
        self.assert_valid_scene(scene)
        
        # check that rendering libraries are installed, if not, prompt user to install optional dependencies
        import pyrender

        if 'PYOPENGL_PLATFORM' not in os.environ or os.environ['PYOPENGL_PLATFORM'] not in ["egl", "osmesa"]:
            logging.warn("You should set the PYOPENGL_PLATFORM environment variable before importing pyrender or any other OpenGL library. \n\tSetting PYOPENGL_PLATFORM=`egl`. \n\tSee https://pyrender.readthedocs.io/en/latest/examples/offscreen.html ")
            os.environ['PYOPENGL_PLATFORM'] = "egl"
            reload(pyrender)
        
        with DisableLogger():
            img = self.get_img(scene, self.camera.images[0])
        
        height, width = img.shape[:2]
        r = pyrender.OffscreenRenderer(width, height)
        
        scene_render = pyrender.Scene(bg_color=[0, 0, 0, 0])
        
        K = self.cam_K
        c2w = CameraFramework.convert(self.cam_c2w, CameraFramework.OPENCV, CameraFramework.OPENGL)
        icamera = pyrender.IntrinsicsCamera(fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2], zfar=10000)
        scene_render.add(icamera, pose=c2w)
        logging.debug(f"K: {K}")
        logging.debug(f"c2w: {c2w}")

        node_dict = {}
        for part, group in itertools.groupby(self.objects, operator.itemgetter(0)):
            try:
                part_trimesh = self.get_mesh(part)
            except:
                logging.warn(f"No mesh found for {part}, skipping mask renders")
                continue
            
            mesh = pyrender.Mesh.from_trimesh(part_trimesh)

            for _, instance in group:
                o2c = self.o2c.sel(scene=scene, part=part, instance=instance).to_numpy()
                node = pyrender.Node(mesh=mesh, matrix=o2c)
                node_dict[(part, instance)] = node
                scene_render.add_node(node)

        seg_masks_by_object = {}
        for key in self.objects:
            if key not in node_dict:
                logging.warn(f"No mask created for {key}.")
                continue
            node = node_dict[key]
            nodes = {node: 255}
            seg_mask = r.render(scene_render, pyrender.constants.RenderFlags.SEG, nodes)
            seg_masks_by_object[key] = seg_mask[0]
        
        return seg_masks_by_object
    
    