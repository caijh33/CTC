I3D_TCC_Bilinear
======
Overview
------
This is a PyTorch code for action recognition reported in the paper “The Combination of Temporal-Channels correlation information and Bilinear feature for action recognition”. Please also refer to [here](https://github.com/hassony2/kinetics_i3d_pytorch) for pretrained models and details about I3D.

Requisites
----------
## Software
*	Ubuntu 16.04
*	Python 3.6
*	PyTorch 0.4
*	Cuda 9.0
*	Cudnn 7.5
## Hardware
*	GTX 1080

Runing the code
--------------
### 1. clone this repository
$ git clone https://github.com/caijh33/I3D_TCC_Bilinear

### 2. train on UCF101 on RGB data and flow data
##### finetune on split1 of RGB data of UCF101
```python I3D_TCC_Bilinear/scripts/ucf101/run_ucf101_i3d_rgb.sh```
##### finetune on split1 of flow data of UCF101
```python I3D_TCC_Bilinear/scripts/ucf101/run_ucf101_i3d_flow.sh```



### 3. test on UCF101 on RGB data and flow data
After you have trained the model, you can run the test procedure. you can run testing you trained models on RGB data and flow data using below commands:

```python test_i3d.py --train_list data/ucf101_rgb_train_split_1.txt \
 --val_list data/ucf101_rgb_val_split_1.txt --data ucf101 --model rgb\
--weights checkpoints/hmdb51/73.202_rgb_model_best.pth.tar \
--save_scores test_output/ --test_clips 10\ ```

```python test_i3d.py --train_list data/ucf101_flow_train_split_1.txt \
 --val_list data/ucf101_flow_val_split_1.txt --data ucf101 --model flow\
--weights checkpoints/hmdb51/73.202_flow_model_best.pth.tar \
--save_scores test_output/ --test_clips 10\ ```

Other models use the same training and testing methods.






