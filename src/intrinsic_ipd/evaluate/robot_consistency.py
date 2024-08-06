from ..ipd import IPDataset

import numpy as np

class MaximumVertexDistance():

    def __init__(self) -> None:
        super().__init__()

    def measure_accuracy(
        self, 
        pose_true: np.ndarray, 
        pose_pred: np.ndarray, 
        vertices: list[np.ndarray], 
        **kwargs
    ) -> float:
        """
        Given 
        Args:
            pose_true (np.ndarray): True object pose relative to the camera [4, 4] 
            pose_pred (np.ndarray): Estimated object pose relative to the camera [4, 4]
            vertices (np.ndarray): Set of k 3D vertices on the object mesh [k, 3]
        """        
        vertices = np.c_[vertices, np.ones(len(vertices))]
        return np.max(np.linalg.norm(pose_true @ vertices.T - pose_pred @ vertices.T, axis=0))

class RobotConsistency():
    def __init__(
        self,
        dataset: IPDataset,
        *args,
        **kwargs
    ):
        super().__init__(self, *args, **kwargs)

    def evaluate(
        self,
        c2o_preds_by_scene:list[list[np.ndarray]],
        pred_names_by_scene:list[list[str]],
        g2r_by_scene:list[np.ndarray],
        *args,
        **kwargs
    ):
        """_summary_

        Args:
            c2o_preds_by_scene (list[list[np.ndarray]]): A list of poses (Tco_hat) for predicted objects for each scene [[[4, 4] x num obj] x num scenes]
            pred_names_by_scene (list[list[str]]):  Object names corresponding predictions in c2o_preds_by_scene [[num obj] x num scenes]
            r2c (np.ndarray): Transform from robot to camera, i.e. Tcr. [4, 4]
            g2r_by_scene (list[np.ndarray]): Transforms from gripper to robot for each scene [[4,4] x num scenes]
        """

        # Calc o2g_hat for all objects for all scenes
        g2r_s = np.array(g2r_by_scene)
        c2o_hats_s = np.array(c2o_preds_by_scene) #maybe run into error with uneven lists of objects
        o2g_preds_s = g2r_s @ r2c @ c2o_hats_s
        o2g_preds_by_scene = o2g_preds_s.tolist() 

    
    
    def match_objects_across_scenes(
        self,
        # predictions
        o2g_preds_by_scene:list[list[np.ndarray]],
        obj_names_by_scene:list[list[str]],
        # ground truth
        o2g_truths_by_scene:list[list[np.ndarray]],
        truth_names:list[str],
        truth_ids:list[int],
        # matching params
        threshold:float,
        *args,
        **kwargs
    ) -> list[list[np.ndarray]]:
        """_summary_

        Args:
            o2g_preds_by_scene (list[list[np.ndarray]]): A list of lists, where each inner list represents a scene. Each scene contains a list of predicted object poses (4x4 matrices) relative to the gripper frame.
            pred_names_by_scene (list[list[str]]): A list of lists, where each inner list contains the names of the objects corresponding to the predicted poses in the respective scene.
            o2g_truths_by_scene (list[list[np.ndarray]]): A list of lists, similar to o2g_preds_by_scene, but containing the ground truth object poses for each scene.
            truth_names (list[str]): A list of the names of all objects present in any given scene of o2g_truths_by_scene.
            truth_id(list[int]]): A list of unique identifiers for each object in any given scene of o2g_truths_by_scene.
            threshold(float): A threshold for determining if a predicted pose is considered a match to a ground truth pose.

        Returns:
            list[list[np.ndarray]]: A list of lists, where each inner list represents a specific object. Each object's list contains the predicted poses (4x4 matrices) for that object across all scenes. If the object wasn't detected in a scene, the corresponding pose will be None.
        """
        # For each scene, match the predictions to the ground truth according to whichever object is closest and beneath given threshold
            # for each sub list of o2g_preds_by_scene
        # Sanity check that the matched prediction's object name is the same as the ground truth pose's object name
        # If no match found or if conflicting matches found, assign no match.ß
        # Should get o2g_matches_by_scene
        # Then reshape to get o2g_scenes_by_obj_id 
        
        o2g_matches_by_scene = []  # Initialize a list to store matched poses for each scene
        for scene_idx, (o2g_preds, obj_names, o2g_truths) in enumerate(zip(o2g_preds_by_scene, obj_names_by_scene, o2g_truths_by_scene)):
            matched_poses = []  # Initialize a list to store matched poses for the current scene
            for truth_idx, (truth_pose, truth_name, truth_id) in enumerate(zip(o2g_truths, truth_names, truth_ids)):
                # Find the closest predicted pose to the ground truth pose
                closest_pred_idx = np.argmin(np.linalg.norm(o2g_preds - truth_pose, axis=(1, 2)))
                closest_pred_pose = o2g_preds[closest_pred_idx]
                closest_pred_name = obj_names[closest_pred_idx]

                # Check if the closest prediction matches the ground truth name and if it's within a threshold
                if closest_pred_name == truth_name and np.linalg.norm(closest_pred_pose - truth_pose, axis=(1, 2)) < threshold:
                    matched_poses.append(closest_pred_pose)
                else:
                    matched_poses.append(None)  # Assign None if no match or conflicting matches

            o2g_matches_by_scene.append(matched_poses)  # Store matched poses for the current scene

        # Reshape the matched poses to group by object ID
        o2g_scenes_by_obj_id = [[] for _ in range(len(truth_ids))]  # Initialize a list for each object
        for scene_idx, matched_poses in enumerate(o2g_matches_by_scene):
            for obj_idx, matched_pose in enumerate(matched_poses):
                o2g_scenes_by_obj_id[obj_idx].append(matched_pose)  # Add the matched pose to the object's list

        return o2g_scenes_by_obj_id
        