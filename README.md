# IMUWiFine

Indoor localization based on IMU and WiFi Signals. 


# Setup and Requirements

Our models are developed using the Pytorch framework, please be sure to install the framework from https://pytorch.org/  


# Downloading the Dataset

The dataset can be downloaded from issai.nu.edu.kz/imuwifine. The dataset consists of `train`, `test` and `validation` folders. The path to the downloaded folders should be specified in `train.py` 

```python
  train_data_path = 'path to train folder'
  val_data_path = 'path to validation folder'
  test_data_path = 'path to test folder'
```


# Training

Before starting the training, please be sure to name the experiment in `train.py` ```python experiment = 'Name of the Experiment```, doing so will differentiate between instances of training.
