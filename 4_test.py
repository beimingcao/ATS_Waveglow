import time
import yaml
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from models import MyLSTM
from models import RegressionLoss
from models import save_model
from measures import MCD
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import ndimage

def test_LSTM(test_SPK, test_dataset, exp_output_folder, args):
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    save_output = config['testing_setup']['save_output']
    synthesis_samples = config['testing_setup']['synthesis_samples']
    metric = MCD()

    test_data = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False, drop_last=False)
    test_out_folder = os.path.join(exp_output_folder, 'testing')
    model_out_folder = os.path.join(exp_output_folder, 'trained_models')
    if not os.path.exists(test_out_folder):
        os.makedirs(test_out_folder)
   
    SPK_model_path = os.path.join(model_out_folder)
    model_path = os.path.join(SPK_model_path, test_SPK + '_lstm')
    model = MyLSTM()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    acc_vals = []
    for file_id, x, y in test_data:
        x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)

        h, c = model.init_hidden(x)
        with torch.no_grad():
            y_head = model(x, h, c)

        y_pt = y_head.squeeze(0).T

        if save_output == True:
            outpath = os.path.join(test_out_folder, test_SPK)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            torch.save(y_pt, os.path.join(outpath, file_id[0] + '.pt'))

        acc_vals.append(metric(y.squeeze(0), y_head.squeeze(0)))
    avg_vacc = sum(acc_vals) / len(acc_vals)

    results_out_folder = os.path.join(exp_output_folder, 'RESULTS')
    if not os.path.exists(results_out_folder):
        os.makedirs(results_out_folder)

    results = os.path.join(results_out_folder, test_SPK + '_results.txt')
    with open(results, 'w') as r:
        print('MCD = %0.3f' % avg_vacc, file = r)
    r.close()
       



if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/ATS_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)

    data_path = os.path.join(args.buff_dir, 'data_CV')
    SPK_list = config['data_setup']['spk_list']

    for test_SPK in SPK_list:
        data_path_SPK = os.path.join(data_path, test_SPK)
        te = open(os.path.join(data_path_SPK, 'test_data.pkl'), 'rb')
        test_dataset = pickle.load(te)

        test_LSTM(test_SPK, test_dataset, args.buff_dir, args)


