
# Industrial Plentopic Dataset

![alt text](assets/teaser.png)

Accepted at CVPR 2024!

# Abstract
6DoF Pose estimation has been gaining increased importance in vision for over a decade, however it does not yet meet the reliability and accuracy standards for mass deployment in industrial robotics. To this effect, we present the Industrial Plenoptic Dataset (IPD): the first dataset and evaluation method for the co-evaluation of cameras, HDR, and algorithms targeted at reliable, high-accuracy industrial automation. Specifically, we capture 2,300 physical scenes of 22 industrial parts covering a $1m\times 1m\times 0.5m$ working volume, resulting in over 100,000 distinct object views. Each scene is captured with 13 well-calibrated multi-modal cameras including polarization and high-resolution structured light. In terms of lighting, we capture each scene at 4 exposures and in 3 challenging lighting conditions ranging from 100 lux to 100,000 lux. We also present, validate, and analyze robot consistency, an evaluation method targeted at scalable, high accuracy evaluation. We hope that vision systems that succeed on this dataset will have direct industry impact. 

## This Repo

- [x] March 29th: Dataset Released!
- [ ] April 15th: Scanned CAD Available
- [ ] May 15th: Code for Robot Consistency Evaluation Method
- [ ] June 1st: Code for downloading and visualization data
- [ ] June 15th: Leaderboard for submitting results on test images

## Dataset
![alt text](assets/dataset.png)

In the repo you can find the evaluation dataset as well as links to relevant cad models

### Evaluation Data

| Dataset ID                  | Parts                                                                                               | Background | Cameras                                                                                                                              |   |
|-----------------------------|-----------------------------------------------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------------------------------------------------|---|
| dataset_2023-10-18_13-38-42 | ['gear1', 'u_bolt', 'pegboard_basket']                                                              | basket     | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-19_08-07-22 | ['gear2', 't_bracket', 'corner_bracket6', 'tote_basket']                                            | basket     | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-19_10-12-18 | ['square_bracket', 'single_pinch_clamp', 'corner_bracket1', 'pegboard_basket']                      | basket     | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-19_12-12-27 | ['l_bracket', 'handrail_bracket', 'corner_bracket2', 'corner_bracket3', 'tote_basket']              | basket     | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-19_14-01-44 | ['corner_bracket', 'door_roller', 'corner_bracket4', 'pegboard_basket']                             | basket     | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-20_08-55-33 | ['oblong_float', 'corner_bracket5', 'tote_basket']                                                  | basket     | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-20_10-44-46 | ['wraparound_bracket', 'hex_manifold', 'elbow_connector', 'pegboard_basket']                        | basket     | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-23_10-47-53 | ['helical_insert', 'corner_bracket0', 'tote_basket']                                                | basket     | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-23_12-35-16 | ['pipe_fitting_unthreaded', 'pull_handle', 'pegboard_basket']                                       | basket     | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-23_14-26-45 | ['access_port', 'corner_bracket5', 'tote_basket']                                                   | basket     | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-24_10-03-05 | ['strut_channel', 'load_securing_track', 'corner_bracket4', 'pegboard_basket']                      | basket     | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-24_11-51-40 | ['gear2', 'hex_manifold']                                                                           | dark shiny		 | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-24_13-54-41 | ['gear1', 'wraparound_bracket']                                                                     | dark shiny		 | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-24_15-44-07 | ['square_bracket', 'handrail_bracket']                                                              | dark shiny		 | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-25_08-08-32 | ['l_bracket', 'helical_insert', 'corner_bracket0']                                                  | dark shiny		 | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-25_10-17-15 | ['t_bracket', 'corner_bracket', 'access_port', 'corner_bracket4']                                   | dark shiny		 | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-25_12-13-25 | ['oblong_float', 'pipe_fitting_unthreaded', 'corner_bracket5']                                      | dark shiny		 | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-25_14-17-56 | ['u_bolt', 'elbow_connector', 'corner_bracket2', 'corner_bracket3']                                 | dark shiny		 | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-25_20-33-27 | ['load_securing_track', 'pull_handle', 'corner_bracket1', 'corner_bracket6']                        | dark shiny		 | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-26_10-03-42 | ['door_roller', 'single_pinch_clamp']                                                               | dark shiny		 | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-26_12-01-45 | ['gear2', 'hex_manifold', 'pull_handle']                                                            | textured   | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-26_14-01-04 | ['gear1', 'wraparound_bracket', 'corner_bracket0', 'handrail_bracket', 'corner_bracket6']           | textured   | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-26_22-07-16 | ['access_port', 'elbow_connector', 'corner_bracket1', 'helical_insert']                             | textured   | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-27_08-32-33 | ['square_bracket', 'l_bracket', 't_bracket', 'corner_bracket', 'door_roller']                       | textured   | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-27_10-55-52 | ['load_securing_track', 'corner_bracket2', 'corner_bracket3', 'corner_bracket4', 'corner_bracket5'] | textured   | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |
| dataset_2023-10-27_12-59-16 | ['u_bolt', 'pipe_fitting_unthreaded', 'single_pinch_clamp', 'oblong_float']                         | textured   | [Basler-LR](https://github.com), [Basler-HR](https://github.com), [PhotoNeo](https://github.com), [Flir-Polar](https://github.com),  |   |

> Dataset is in BOP format
```bash
dataset_id/camera/
--- calibration.pkl
--- robot_poses.json
--- ground_truth.json
--- scene_0/
------ cam_id_1/
--------- img.png
------ cam_id_2/
--------- img.png
------ cam_id_3/
--------- img.png
--- scene_1/ ... n
```

### CAD Models - Preferred Method
We have created our own scanned versions of the CAD for each part released under our CC-BY Non-Commercial License. Those are available here. 
| Part Name | CAD Link | 
| --------- | -------- |
| Gear | [Link](https://github.com) |

### CAD Models - Paper Method
For the results in the paper we used the CAD models from mcmaster-carr's website after purchasing the part. These you may acquire yourself from them as per their license terms. 

| Part Name | Part ID Number | 
| --------- | -------- |
| Gear | [89234015-1](https://github.com) |
