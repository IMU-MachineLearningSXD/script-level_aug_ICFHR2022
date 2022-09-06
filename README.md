# Script-Level Word Sample Augmentation for Few-shot Handwritten Text Recognition  
This project provides a script-level augmentation method for handwritten text recognition  
Accepted at ICFHR2022  
## Table of Contents
[Overview](https://github.com/IMU-MachineLearningSXD/script-level_aug_ICFHR2022/edit/main/README.md#overview)
## Overview
![image](https://github.com/IMU-MachineLearningSXD/script-level_aug_ICFHR2022/blob/main/dst/flow.jpg)
At first, we extract the skeleton of each handwritten word and segment the whole word into several components with the skeleton and eight-neighborhood information. At second, we move the three control points of each component according a empirical formula, since each component is controlled by Bézier control points and making script variation is based on the control point movement. Finally, we transform new script from the control points after movement by quadratic Bézier curves and assemble them together to form a new word sample.

