from collections import defaultdict
from .reader import IPDReader
from .matcher import PoseMatcher
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union
import logging

class Evaluator:
    def __init__(self, 
                 reader: IPDReader,
                #  dist_thresh_by_part: dict[str, float] = {},
                #  default_dist_thresh: float = 0.01,
                 ):
        self.reader = reader
    

    def measure_ground_truth_accuracy(self, 
                                      o2c_pred:xr.DataArray,
                                      o2c_gt:xr.DataArray = None,
                                      metric:str='mvd') -> xr.DataArray:
        """ Measures accuracy of predicted poses against ground truth poses.

        Args:
            o2c_pred (xr.DataArray): Predicted poses (same dims and coords as self.reader.o2c)
            o2c_gt (xr.DataArray, optional): Ground truth poses. Defaults to None, which will use self.reader.o2c.
            metric (str, optional): 'mvd' or 'add'. Defaults to 'mvd'.

        Returns:
            xr.DataArray: Accuracy of predicted poses for each object.
        """
        assert metric in ['mvd', 'add'], "unknown metric for pose accuracy, please pick one of ['mvd', 'add']"

        if o2c_gt is None:
            o2c_gt = self.reader.o2c

        if o2c_pred is None:
            o2c_pred = self.matched_o2c

        # Remove symmetry
        o2c_gt = self.reader.remove_symmetry_xarray(o2c_gt)
        o2c_pred = self.reader.remove_symmetry_xarray(o2c_pred)

        pose_diff = o2c_gt - o2c_pred

        # Setup function to compute vertex distance
        def vertex_distance(pose_diff:np.ndarray, vertices:np.ndarray, metric:str):
                logging.debug(f"received {type(pose_diff)} shape: {pose_diff.shape}")
                logging.debug(f'vertices.shape: {vertices.shape}')
                result = np.linalg.norm(pose_diff@vertices.T, axis=-2)
                logging.debug(f"result.shape: {result.shape}")
                return result
        
        data_arrays = []
        # Compute accuracy for each part
        for part, group in pose_diff.groupby("part"):
            # Get vertices for part
            try:
                part_trimesh = self.reader.get_mesh(part)
            except:
                logging.debug(f"No mesh found for {part}, skipping.")
                continue

            vertices = part_trimesh.vertices # shape [n, 3]
            vertices = np.c_[vertices, np.ones(len(vertices))]  # shape [n, 4]

            # Compute vertex distance for each instance
            distance = xr.apply_ufunc(vertex_distance, group,  
                    kwargs={
                        'vertices' : vertices,
                        'metric' : metric
                        },
                    input_core_dims=[
                        ["scene", "object", "transform_major", "transform_minor"]
                    ],
                    output_core_dims=[["scene", "object", "vertex_distance"]],
                    )                                  # shape [scenes, instances, n]
            
            # Compute accuracy by mean or max over vertices
            if metric == 'mvd':
                accuracy = distance.max(dim='vertex_distance', skipna=True)  # shape [scenes, instances]
            elif metric == 'add':
                accuracy = distance.mean(dim='vertex_distance', skipna=True) # shape [scenes, instances]
            
            # Compute accuracy by mean over scenes
            accuracy = accuracy.mean(dim='scene', skipna=True)  # shape [instances]
            data_arrays.append(accuracy)

        return xr.concat(data_arrays, dim='object')
    
    def measure_robot_consistency(self, 
                                  o2c_pred:xr.DataArray,
                                  metric:str='mvd') -> xr.DataArray:
        """Measures robot consistency of predicted poses for an object across scenes.

        Args:
            o2c_pred (xr.DataArray): Predicted poses (same dims and coords as self.reader.o2c)
            metric (str, optional): ['mvd', 'add']. Defaults to 'mvd'.

        Returns:
            xr.DataArray: Robot consistency metrics for each object.
        """
        
        # Get predicted object poses in gripper frame
        c2g = self.reader.c2g
        o2g_pred = xr.apply_ufunc(np.matmul, c2g.broadcast_like(o2c_pred), o2c_pred, 
               input_core_dims=[
                   ["scene", "object", "transform_major", "transform_minor"],
                   ["scene", "object", "transform_major", "transform_minor"]
                ],
               output_core_dims=[["scene", "object", "transform_major", "transform_minor"]])
        
        # Remove symmetry
        o2g_pred = self.reader.remove_symmetry_xarray(o2g_pred)
       
        # Average over scenes to get approximation of ground truth object poses in gripper frame
        # Should have dims: [object, transform_major, transform_minor]
        o2g_star = o2g_pred.mean(dim='scene', skipna=True)
        
        # Get camera to gripper transformations
        g2c = xr.apply_ufunc(np.linalg.inv, c2g,
                             input_core_dims=[["scene", "transform_major", "transform_minor"]],
                             output_core_dims=[["scene", "transform_major", "transform_minor"]])
        
        g2c_b, o2c_star_b = xr.broadcast(g2c, o2g_star)
        # Transform object poses from gripper frame to camera frame
        # This is the robot consistency "ground truth" object poses
        o2c_star = xr.apply_ufunc(np.matmul, g2c_b, o2c_star_b, 
               input_core_dims=[
                   ["scene", "object", "transform_major", "transform_minor"],
                   ["scene", "object", "transform_major", "transform_minor"]
                ],
               output_core_dims=[["scene", "object", "transform_major", "transform_minor"]])

        return self.measure_ground_truth_accuracy(o2c_gt=o2c_star, o2c_pred=o2c_pred, metric=metric)
    
    def recall(self,
               matcher:PoseMatcher)-> xr.DataArray:
        """ Recall is defined by the number of predictions matched to a ground truth instance for a particular part divided by the number of ground truth instances for that part.

        Args:
            matcher (PoseMatcher): Pose matcher containing registered pose predictions.

        Returns:
            xr.DataArray: Recall by part.
        """
        stats = matcher.get_stats()
        return stats.sel(counts='true_positive') / stats.sel(counts='actual_positive')
    
    def precision(self,
                  matcher:PoseMatcher) -> xr.DataArray:
        """ Precision is defined by the number of predictions matched to a ground truth instance for a particular part divided by the number of predictions for that part.

        Args:
            matcher (PoseMatcher): _description_

        Returns:
            xr.DataArray: _description_
        """
        stats = matcher.get_stats()
        return stats.sel(counts='true_positive') / stats.sel(counts='test_positive') 