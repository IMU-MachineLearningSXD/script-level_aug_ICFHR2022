# Script-Level Word Sample Augmentation for Few-shot Handwritten Text Recognition  
This project provides a script-level augmentation method for handwritten text recognition.  
Accepted at ICFHR2022.  
## Table of Contents
[Overview](https://github.com/IMU-MachineLearningSXD/script-level_aug_ICFHR2022#overview)  
[Requirements](https://github.com/IMU-MachineLearningSXD/script-level_aug_ICFHR2022#requirements)  
[Usage](https://github.com/IMU-MachineLearningSXD/script-level_aug_ICFHR2022#usage)  
[Augmented samples](https://github.com/IMU-MachineLearningSXD/script-level_aug_ICFHR2022#augmented-samples)  
[Citation](https://github.com/IMU-MachineLearningSXD/script-level_aug_ICFHR2022#citation)
## Overview
![image](https://github.com/IMU-MachineLearningSXD/script-level_aug_ICFHR2022/blob/main/dst/flow.jpg)
At first, we extract the skeleton of each handwritten word and segment the whole word into several components with the skeleton and eight-neighborhood information. At second, we move the three control points of each component according a empirical formula, since each component is controlled by Bézier control points and making script variation is based on the control point movement. Finally, we transform new script from the control points after movement by quadratic Bézier curves and assemble them together to form a new word sample.  
## Requirements   
- python 3.0+  
- skimage  
- opencv-python  
- numpy  
## Usage
- We provide some handwritten images that can be used directly in the `src` folder.
- We also provide three deformation methods, which are Bezier, Affine and L2A. These methods can be used individually or in combination. The code will default to Bezier curve deformation.
### Run the following code without making any adjustments
    python script_aug.py
### Parameter modification  
| Parameter                 |    Default      | Modified location         |
| :---:                     |    :----:       |         :---:             |
|input                      |./`src`/`0.jpg`  | `script_aug.py` line 142  |
|output                     |./`dst`          | `script_aug.py` line 145  |
|augment_times              |10               | `script_aug.py` line 143  |
|transformation method      |bezier           | `script_aug.py` line 64   |
|control point movement area|0.6              | `script_aug.py` line 14   |
|script_thickness           |2                | `script_aug.py` line 14   |
|background_color           |white            | `script_aug.py` line 108  |
|script_color               |black            | `script_aug.py` line 118  | 

## Augmented samples
![image](https://github.com/IMU-MachineLearningSXD/script-level_aug_ICFHR2022/blob/main/dst/samples.jpg)

## Citation
Waiting for the ICFHR2022 to start. 
