from model import *
import time
from dataset import WifiDataset
from train_wifi_model import BATCH_SIZE, EPOCHS, experiment, DEVICE
import pandas as pd
import numpy as np
from os import sep, makedirs
import glob
from os.path import exists
from torch.utils.data import DataLoader


def repackage_hidden(hidden_states):
    '''
    Wraps hidden states in new Tensors, to detach them from their history.

    Arguments:
    hidden_states -- RNN hidden state

    Returns:
    Hidden states as a tuple
    '''
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states.detach()
    else:
        return tuple(repackage_hidden(h) for h in hidden_states)

def save_model_wifi(model):
    path = experiment + sep + 'model'
    if not exists(path):
        makedirs(path)
    torch.save(model.state_dict(), path + sep + 'model.pt')


def calculate_errors_wifi(type_):
    if type_ == 'test':
        path = experiment + sep + 'outputs' + sep + 'test'
    elif type_ == 'val':
        path = experiment + sep + 'outputs' + sep + 'validation'
    files = glob.glob(path + sep + '*.csv')

    errors_all = []
    error_third_quartile_avg = 0.0
    num_files = 0
    for file in files:
        df = pd.read_csv(file)
        target = df.iloc[:, :3].to_numpy()
        preds = df.iloc[:, 3:].to_numpy()
        errors_trajectory = []
        for p, t in zip(preds, target):
            if p[0] == 0 and p[1] == 0:
                continue
            error_cur = np.linalg.norm(p-t)
            errors_all.append(error_cur)
            errors_trajectory.append(error_cur)

        error_third_quartile = np.percentile(errors_all, 75)
        error_third_quartile_avg += error_third_quartile
        num_files += 1
    errors_all.sort()

    med = np.mean(errors_all)
    error_third_quartile = np.percentile(errors_all, 75)
    error_third_quartile_avg /= num_files
    return med, error_third_quartile, error_third_quartile_avg


def save_predictions_wifi(target, preds, type_, file_name):
    df = pd.DataFrame(np.concatenate((target[:, :3], preds[:, :3]), axis=1), columns=['x', 'y', 'z', 'x_pred', 'y_pred', 'z_pred'])
    if type_ == 'val':
        path_type = 'validation'
    if type_ == 'test':
        path_type = 'test'
    path = experiment + sep + 'outputs' + sep + path_type
    if not exists(path):
        makedirs(path)
    df.to_csv(path + sep + file_name[0], index=False)


def validate_model_wifi(wifi_model, data, type_):
    wifi_model.eval()
    print(experiment)
    for val_trajectory in data:
        with torch.no_grad():
            to_csv_pred = np.zeros((BATCH_SIZE, 3))
            to_csv_target = np.zeros((BATCH_SIZE, 3))
            hidden = wifi_model.init_hidden(BATCH_SIZE)
            for wifi, posi, trajectory in val_trajectory:
                f_name = trajectory
                wifi, posi = wifi.to(DEVICE), posi.to(DEVICE)
                posi_pred, hidden = wifi_model(wifi, hidden)
                posi_pred = posi_pred.squeeze(dim=0)
                posi = posi.squeeze(dim=0)
                preds, posi = posi_pred.cpu().numpy(), posi.cpu().numpy()
                to_csv_pred = np.concatenate((to_csv_pred, preds), axis=0)
                to_csv_target = np.concatenate((to_csv_target, posi), axis=0)

            save_predictions_wifi(to_csv_target, to_csv_pred, type_, f_name)
    errors = calculate_errors_wifi(type_)
    return errors


def train_wifi_model(wifi_model, optimizer, loss_fn, train_data_traj, val_data_traj, scheduler):
    med_val_best = float('inf')
    best_epoch = 0
    print('Started training Wifi Model')

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(EPOCHS):
        print('epoch number', epoch)
        start_time = time.time()
        wifi_model.train()
        train_loss = 0.0
        train_len = 0
        for train_data in train_data_traj:
            hidden = wifi_model.init_hidden(BATCH_SIZE)
            for wifi, posi, _ in train_data:
                wifi, posi = wifi.to(DEVICE), posi.to(DEVICE)
                hidden = repackage_hidden(hidden)
                posi_pred, hidden = wifi_model(wifi, hidden)
                loss = loss_fn(posi, posi_pred)
                train_loss += loss.item()
                train_len += len(wifi)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

        scheduler.step()
        train_loss /= train_len

        wifi_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_len = 0
            for val_data_ in val_data_traj:
                hidden = wifi_model.init_hidden(BATCH_SIZE)
                for wifi, posi, _ in val_data_:
                    wifi, posi = wifi.to(DEVICE), posi.to(DEVICE)
                    hidden = repackage_hidden(hidden)
                    posi_pred, hidden = wifi_model(wifi, hidden)
                    loss = loss_fn(posi_pred, posi)
                    val_loss += loss.item()
                    val_len += len(posi)
            val_loss /= val_len

        med_val, _, _ = validate_model_wifi(wifi_model, val_data_traj, 'val')

        print('epoch : ', epoch)
        print('train loss : ', train_loss, end='\t')
        print('val_loss : ', val_loss, end='\t\t')
        print('val_med : ', med_val, end='\t\t')
        print('lr : ', optimizer.param_groups[0]['lr'], end='')
        if med_val < med_val_best:
            med_val_best = med_val
            save_model_wifi(wifi_model)
            best_epoch = epoch
        epoch_time = (time.time() - start_time) / 60.0
        print('epoch time', epoch_time)
        print('best epoch {}, best val med {}'.format(best_epoch, med_val_best))
        print('experiment:', experiment)

if __name__ == '__main__':
    fusion_model = FusionModel()
    imu_output = torch.rand(32, 150, 3)
    imu_output = torch.sum(imu_output, dim=1)

    wifi_output = torch.rand(32, 150, 3)[:, -1, :]
    model = FusionModel()
    pred = model(imu_output, wifi_output)
    print(pred.shape)