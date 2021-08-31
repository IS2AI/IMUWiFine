# IMUWiFine

Indoor localization based on IMU and WiFi Signals. 
The project aims to enhance the [WiFi based indoor localization](https://ieeexplore.ieee.org/document/9382623) with the signals from Inertial Measurement Units built-in in smartphones. The detailed explanation can be found on paper [](). 

# Setup and Requirements

Our models are developed using the Pytorch framework, please be sure to install the framework from official [Pytorch](https://pytorch.org/) page.  


# Downloading the Dataset

The dataset can be downloaded from [ISSAI](https://issai.nu.edu.kz/imuwifine). The dataset consists of `train`, `test` and `validation` folders. The path to the downloaded folders should be specified in `train.py` 

```python
  train_data_path = 'path to train folder'
  val_data_path = 'path to validation folder'
  test_data_path = 'path to test folder'
```


# Training

Before starting the training, please be sure to name the experiment in `train.py`, doing so will differentiate between instances of training.
```python 
  experiment = 'Name of the Experiment'
```
To train the model run `python train.py` inside IMUWiFine folder. Once the training finished, the script will automatically test the model on `testing set`. 
