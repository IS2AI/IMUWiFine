from utils import *
import glob
EPOCHS = 1200
BATCH_SIZE = WINDOW_SIZE
DEVICE = 'cuda:0'

train_path = 'data/train_wifi'
val_path = 'data/val_wifi'
test_path = 'data/test_wifi'

experiment = '4_lstm_4_relu_layer_512'


train_data_path = glob.glob(train_path + '/*.csv')
val_data_path = glob.glob(val_path + '/*.csv')
test_data_path = glob.glob(test_path + '/*.csv')


def create_dataloader(path):
    dataloaders = []
    i = 0
    for file in path:

        data_set = WifiDataset(file)
        loader = DataLoader(data_set, num_workers=4, drop_last=True)
        dataloaders.append(loader)
        i += 1
    return dataloaders

weight_decay = 0.0
lr = 0.001
momentum = 0.9


if __name__ == '__main__':
    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WIFI_MODEL().to(DEVICE)

    train_data = create_dataloader(train_data_path)
    val_data = create_dataloader(val_data_path)
    test_data = create_dataloader(test_data_path)

    model_name = type(model).__name__
    model_name += 'exp' + experiment + '_epochs' + str(EPOCHS) + '_bs' + str(BATCH_SIZE)
    print(model_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    loss_fn = nn.MSELoss(reduction='sum')

    train_wifi_model(model, optimizer, loss_fn, train_data, val_data, scheduler)
    path = experiment + sep + 'model' + sep + 'model.pt'
    model.load_state_dict(torch.load(path))
    model.eval()
    error_val, third_quartile_error_val, error_third_quartile_avg_val = validate_model_wifi(model, val_data, 'val')
    error_test, third_quartile_error_test, error_third_quartile_avg_test = validate_model_wifi(model, test_data, 'test')
    error_metrics = {}
    error_metrics[model_name] = {
        "med val": error_val,
        "third_quartile_error_val": third_quartile_error_val,
        "error_third_quartile_avg_val": error_third_quartile_avg_val,
        "med test": error_test,
        "third_quartile_error_test": third_quartile_error_test,
        "error_third_quartile_avg_test": error_third_quartile_avg_test
    }

    print(
        "\n{}\nmed val: {:5.2f}\t\tthird quartile val: {:5.2f}\tthird quartile avg val: {:5.2f}\nmed test: {:5.2f}\t\tthird quartile test: {:5.2f}\tthird quartile avg test: {:5.2f}\n".format(
            model_name, error_val, third_quartile_error_val, error_third_quartile_avg_val, error_test,
            third_quartile_error_test, error_third_quartile_avg_test))
