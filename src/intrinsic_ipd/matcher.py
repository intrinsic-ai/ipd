from collections import defaultdict
from typing import Literal
import scipy
import scipy.optimize
import lapsolver
from .reader import IPDReader
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union
import logging

class PoseMatcher:
    def __init__(self, 
                 reader: IPDReader,
                 ordered: bool = False,
                #  dist_thresh_by_part: dict[str, float] = {},
                 match_dist_default_thresh: float = 100,
                 ):
        self.ordered = ordered
        self.reader = reader

        self.matched_o2c = xr.full_like(self.reader.o2c, np.nan)
        self.raw_o2c_dict = defaultdict(dict)
        self.gt_o2c = reader.remove_symmetry_xarray(reader.o2c)

        self.default_thresh = match_dist_default_thresh
    
    def register_poses(self,
            scene:int,
            part:str,
            poses:Union[np.ndarray, list[np.ndarray]],
            mode:Literal["override", "append"]="override",
            )-> None:
        self.reader.assert_valid_scene(scene)

        # Get poses into correct format
        if isinstance(poses, list):
            poses = np.stack(poses)

        assert len(poses.shape) == 3 and poses.shape[-1] == 4 and poses.shape[-2] == 4, "`poses` must be a list or array of 4x4 poses."
        
        # If override, delete all previously raw or matched poses for this scene/part
        if mode == "override":
            self.clear(scene, part)
        
        if self.ordered:
            # Save poses
            self._save_matched_poses(scene, part, poses)
        else:
            # Save poses into self.raw_o2c dict
            if part in self.raw_o2c_dict[scene]:
                self.raw_o2c_dict[scene][part] = np.concatenate((self.raw_o2c[scene][part], poses))
            else:
                self.raw_o2c_dict[scene][part] = poses
            # Match poses and save
            ordered_poses = self._match_poses(scene, part, self.raw_o2c_dict[scene][part], self.default_thresh) # Get ordered poses (nans where no match)
            self.matched_o2c.loc[scene, part] = np.nan #clear previous matches
            self._save_matched_poses(scene, part, ordered_poses) 
    
    def clear(self, scene:int, part:str)->None:
        self.reader.assert_valid_scene(scene)
        self.reader.assert_valid_part(part)

        if part in self.raw_o2c_dict[scene]:
            del self.raw_o2c_dict[scene][part] 

        self.matched_o2c.loc[scene, part] = np.nan
    
    def clear_all(self):
        self.raw_o2c = defaultdict(dict)
        self.matched_o2c = xr.full_like(self.reader.o2c, np.nan)
    
    def get_matched_poses(self):
        return self.matched_o2c.copy()
    
    def _match_poses(
                           self, 
                           scene:int,
                           part:str, 
                           poses:Union[np.ndarray, list[np.ndarray]],
                           thresh:float
                           ) -> None:
        
        # Match poses 
        matches = self._get_match_pairings(scene, part, poses, thresh)

        # Select ordered poses and provide as list to _register_matched_poses
        num_instances = self.gt_o2c.sel(scene=scene, part=part).sizes["instance"]
        ordered_poses = np.full([num_instances, 4, 4], np.nan)

        # Fill in ordered poses using matches
        for match in matches:
            pred_i, true_i = match
            ordered_poses[true_i, :, :] = poses[pred_i, :, :]
        
        return ordered_poses
    
    def _save_matched_poses(self, 
                               scene:int,
                               part:str, 
                               poses:np.ndarray
                               ):
        """Saves matched poses to self.prematched_o2c.

        Args:
            scene (int): _description_
            part (str): _description_
            poses (Union[np.ndarray, list[np.ndarray]]): _description_
        """
        #find the unmatched instances
        matched_poses = self.matched_o2c.sel(scene=scene, part=part)
        unmatched_poses = matched_poses.where(matched_poses.isnull(), drop=True)
        
        #assert that the number of poses is less than or equal to the number of unmatched instances
        assert unmatched_poses.sizes["instance"] >= poses.shape[0], f"Number of ordered poses ({poses.shape[0]}) saved must be less or equal to number of instances without a saved pose ({unmatched_poses.sizes['instance']})."
        
        #insert the poses where there are empty spots
        self.matched_o2c.loc[{
            "scene": scene,
            "object": [(part, i) for i in unmatched_poses["instance"].data[:poses.shape[0]]],
        }] = poses
        
    def _get_match_pairings(self, 
                          scene:int,
                          part:str,
                          poses:np.ndarray,
                          thresh:float,
                          ):

        pred_o2c = self.reader.remove_symmetry(part, poses)
        pred_o2c = xr.DataArray(pred_o2c,
                               dims=["pred_instance", "transform_major", "transform_minor"],
                               coords={
                                   "pred_instance": range(pred_o2c.shape[0]),
                                   "transform_major": [0, 1, 2, 3],
                                   "transform_minor": [0, 1, 2, 3]
                               }).assign_coords({"scene": scene})

        true_o2c = self.gt_o2c.sel(scene=scene, part=part)

        def translation_distance(pred_o2c, true_o2c):
            logging.debug(f"received {type(pred_o2c), type(true_o2c)} shape: {pred_o2c.shape, true_o2c.shape}")
            translation_diff = pred_o2c[:, np.newaxis, :3, 3] - true_o2c[np.newaxis, :, :3, 3]
            result = np.linalg.norm(translation_diff, axis=-1)
            logging.debug(f"result.shape: {result.shape}")
            return result

        pose_distances = xr.apply_ufunc(translation_distance, pred_o2c, true_o2c, 
                    input_core_dims=[
                        ["pred_instance", "transform_major", "transform_minor"], 
                        ["instance", "transform_major", "transform_minor"]
                    ],
                       output_core_dims=[["pred_instance", "instance"]],
                    )
        
        pose_distances_masked = pose_distances.where(pose_distances < thresh, np.nan)

        logging.debug(f"Pose distances:\n {pose_distances}")
        logging.debug(f"Pose masked:\n {pose_distances_masked}")
        
        def hungarian(matrix, raw_instances, instances):
            raw_ind, true_ind = lapsolver.solve_dense(matrix)
            result = [(raw_instances[i], instances[j]) for i, j in zip(raw_ind, true_ind)]
            return result
        
        # returns a list of pairings from raw to true instances
        return hungarian(pose_distances_masked, pose_distances_masked['pred_instance'].data, pose_distances_masked['instance'].data)
    