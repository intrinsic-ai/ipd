![alt text](assets/image.png)

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

Dataset download is available [here](Dataset.md)
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

### Parts used
We purchased all physical parts from McMaster-Carr's website. We give detailed purchase instructions [here](Parts.md)

## License

All dataset, code, and models available in this repository are given under the CC-BY NC SA license, and are intended for Non-Commercial use only. 