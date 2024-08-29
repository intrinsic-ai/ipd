from . import IPDReader, IPDImage, CameraFramework
import pyrender
import numpy as np
import cv2
import logging
import xarray as xr
import os
from importlib import reload


def render_scene(
        reader:IPDReader, 
        scene:int,
        image_type:IPDImage=None,
        poses:xr.DataArray=None,
        image:np.ndarray=None,
    ):
    """
    Renders the scene with part meshes and poses overlaid on the RGB image.

    Args:
        reader (IPDReader): An instance of the IPDReader class.
        scene (int): ID of the scene.
        image (numpy.ndarray, optional): RGB image to overlay the scene on. 
                                        If None, the image is loaded from the dataset.

    Returns:
        numpy.ndarray: The rendered image with the scene overlaid.
    """

    if 'PYOPENGL_PLATFORM' not in os.environ or os.environ['PYOPENGL_PLATFORM'] not in ["egl", "osmesa"]:
        logging.warn("Set PYOPENGL_PLATFORM environment variable before importing pyrender or any other OpenGL library. \n\tSetting to PYOPENGL_PLATFORM=`egl`. \n\tSee https://pyrender.readthedocs.io/en/latest/examples/offscreen.html ")
        os.environ['PYOPENGL_PLATFORM'] = "egl"
        reload(pyrender)
    
    # Render ground truth poses if poses not provided
    if poses is None:
        poses = reader.o2c
    poses = poses.sel(scene=scene)
    
    # Load the image if not provided
    if image is None:
        if image_type:
            image = reader.get_img(scene, image_type=image_type)
        else:
            image = reader.get_img(scene, image_type=reader.camera.images[0])
    height, width = image.shape[:2]

    # Create a scene
    scene = scene = pyrender.Scene(bg_color=[255, 255, 255, 0])

    # Get camera intrinsics and pose
    K = reader.cam_K
    c2w = CameraFramework.convert(reader.cam_c2w, CameraFramework.OPENCV, CameraFramework.OPENGL)
    icamera = pyrender.IntrinsicsCamera(fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2], zfar=10000)
    scene.add(icamera, pose=c2w)
    logging.debug(f'\nK = \n{K}')
    logging.debug(f'\nc2w = \n{c2w}')

    
    for part, group in poses.groupby("part"):
        try:
            part_trimesh = reader.get_mesh(part)
        except:
            logging.warn(f"No mesh found for {part}, skipping part render")
            continue
        mesh = pyrender.Mesh.from_trimesh(part_trimesh)
        for pose in group:
            logging.debug(f"object {part} pose shape is {pose.shape}")
            scene.add(mesh, pose=pose)

    # Render the scene
    r = pyrender.OffscreenRenderer(width, height)

    rendered_image, depth = r.render(scene)
    
    # Overlay the rendered scene on the original image
    alpha = 0.3  # Transparency of the overlay
    overlay = cv2.addWeighted(image, 1 - alpha, rendered_image, alpha, 0)
    
    return image, rendered_image, overlay