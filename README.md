# IMUWiFine

This repository contains the source code for "End-to-End Sequential Indoor Localization Using Smartphone Inertial Sensors and WiFi" paper. 

The source code implements the end-to-end sequential indoor localization architecture based on the stack of ReLU, LSTM, and regression layers.
Proposed indoor localization architecture takes in WiFi received signal strength indicators (RSSI) and inertial mearurement unit (IMU) readings, and outputs estimated (*x*,*y*,*z*) position.
The detailed description of the architecture is given in the paper. 

# Setup and Requirements

Our code is based on PyTorch framework, please make sure to install the framework from the official [Pytorch](https://pytorch.org/) page.  


# Downloading the Dataset

The dataset can be downloaded from [ISSAI](https://issai.nu.edu.kz/imuwifine). The dataset consists of `train`, `test` and `validation` folders. The paths to the downloaded folders should be specified in `train.py` 

```python
  train_data_path = 'path to train folder'
  val_data_path = 'path to validation folder'
  test_data_path = 'path to test folder'
```


# Training

Before starting the training, please make sure to name the experiment in `train.py`, doing so will help to differentiate between different instances of training.
```python 
  experiment = 'Name of the Experiment'
```
To train a model run `python train.py` inside IMUWiFine folder. Upon completion of the training procedure, the script will automatically evaluate the model on the testing set. 
