# Tactile-Gym Servo Control
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)


**Pose-Based Tactile Servoing**: [Video](https://www.youtube.com/watch?v=12-DJeRcfn0)&nbsp;&nbsp;•&nbsp;&nbsp;[Paper](https://ieeexplore.ieee.org/document/9502718)

**Tactile Gym 2.0**: [Project Website](https://sites.google.com/my.bristol.ac.uk/tactilegym2/home) &nbsp;&nbsp;•&nbsp;&nbsp;[Paper](https://ieeexplore.ieee.org/abstract/document/9847020)

**Tactile Gym 1.0**: [Project Website](https://sites.google.com/my.bristol.ac.uk/tactile-gym-sim2real/home) &nbsp;&nbsp;•&nbsp;&nbsp;[Paper](http://arxiv.org/abs/2106.08796)

This repo contains a simulated implementation of the "*Pose-Based Tactile Servoing: Controlled Soft Touch Using Deep Learning*" [paper](https://ieeexplore.ieee.org/document/9502718). The data collection, supervised deep learning, and servo control procedures are implemented using the open-sourced [Tactile Gym](https://github.com/ac-93/tactile_gym) simulation platform.

There are four main tasks sharing the same underlying data collection, learning and servo control methods. These are **Surface Following 3D**, **Edge Following 2D**, **Edge Following 3D** and **Edge Following 5D**.

<p align="center">
   <img width="256" src="example_videos/surface_3d_saddle.gif" title="Surface Following 3D."> &nbsp;&nbsp;&nbsp;&nbsp;
  <img width="256" src="example_videos/edge_2d_circle.gif" title="Edge Following 2D."> &nbsp;&nbsp;&nbsp;&nbsp;<br>
  <img width="256" src="example_videos/edge_3d_saddle.gif" title="Edge Following 3D."> &nbsp;&nbsp;&nbsp;&nbsp;
  <img width="256" src="example_videos/edge_5d_saddle.gif" title="Edge Following 5D."> &nbsp;&nbsp;&nbsp;&nbsp;
</p>


### Content ###
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Learning](#learning)
- [Servo Control](#servo-control)
- [Bibtex](#bibtex)


### Installation ###

Install Tactile-Gym
```
git clone https://github.com/ac-93/tactile_gym.git
cd tactile_gym
python setup.py install
```

Install Tactile-Gym Servo Control
```
git clone https://github.com/ac-93/tactile_gym_servo_control.git
cd tactile_gym_servo_control
pip install -e .
```

### Data Collection ###

Data can be quickly generated and gathered in simulation.

To gather a small amount of example data with visualisation enabled run

```
python data_collection/collect_data.py -t task_name
```

where task_name is selected from ```[surface_3d edge_2d edge_3d edge_5d]```. If multiple tasks are input they will be executed in the order of input.

To collect a full training and validation sets run

(**WARNING** - This will gather 28,000 images - approximately 100mb)

```
python data_collection/collect_train_and_val_sets.py -t task_name
```

This can be generated significantly faster with rendering and GUI disabled on Ubuntu however a bug for pybullet on Windows causes a crash during collection in this case. The GUI should be enabled if using a Windows machine for data collection.

### Learning ###

This section implements supervised deep learning methods for predicting the pose of the tactile sensors based on the tactile image gathered during data collection.  Pose is encoded and decoded for accurate NN prediction, this uses normalisation for position and  sine/cosine encoding for rotation. Details of this can be found in ```learning/learning_utils.py```.

Image processing and augmentations are used for more robust learning. To visualise the effects of these run

```
python learning/demo_image_generation.py -t task_name
```

To train a CNN for pose prediction run

```
python learning/train_cnn.py -t task_name -d device_name
```
This will overwrite the pretrained models used for demonstrations.

The task, learning and image processing parameters are set within the code. For efficient learning, parameters may need to be tweaked depending on your setup.

A learned model can be evaluated by running

```
python learning/test_cnn.py -t task_name -d device_name
```

### Servo Control ###

Demonstration files are provided for all tasks in the example directory. These use pretrained models included in the repo, training your own models will overwrite these pretrained models. The main logic for servo control is provided in ```servo_control/servo_control.py```.

To demonstrate servo control, from the base directory run

```
python servo_control/servo_control.py -t task_name -d device_name
```
The task can be selected by adjusting the code.

### Bibtex ###

Pose-Based Tactile Servoing
```
@InProceedings{Lepora2021PBTS,
  author={Lepora, Nathan F. and Lloyd, John},
  journal={IEEE Robotics & Automation Magazine},
  title={Pose-Based Tactile Servoing: Controlled Soft Touch Using Deep Learning},
  year={2021},
  volume={28},
  number={4},
  pages={43-55},
  doi={10.1109/MRA.2021.3096141}
  }

```
Tactile Gym 2.0
```
@InProceedings{lin2022tactilegym2,
     title={Tactile Gym 2.0: Sim-to-real Deep Reinforcement Learning for Comparing Low-cost High-Resolution Robot Touch},
     author={Yijiong Lin and John Lloyd and Alex Church and Nathan F. Lepora},
     journal={IEEE Robotics and Automation Letters},
     year={2022},
     volume={7},
     number={4},
     pages={10754-10761},
     editor={R. Liu A.Banerjee},
     series={Proceedings of Machine Learning Research},
     month={August},
     publisher={IEEE},
     doi={10.1109/LRA.2022.3195195}}
     url={https://ieeexplore.ieee.org/abstract/document/9847020},
}
```

Tactile Gym 1.0
```
@InProceedings{church2021optical,
     title={Tactile Sim-to-Real Policy Transfer via Real-to-Sim Image Translation},
     author={Church, Alex and Lloyd, John and Hadsell, Raia and Lepora, Nathan F.},
     booktitle={Proceedings of the 5th Conference on Robot Learning},
     year={2022},
     editor={Faust, Aleksandra and Hsu, David and Neumann, Gerhard},
     volume={164},
     series={Proceedings of Machine Learning Research},
     month={08--11 Nov},
     publisher={PMLR},
     pdf={https://proceedings.mlr.press/v164/church22a/church22a.pdf},
     url={https://proceedings.mlr.press/v164/church22a.html},
}
```
