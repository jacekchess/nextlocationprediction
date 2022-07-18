from pathlib import Path
import pandas as pd
import numpy as np
import torch

def load_sensible(peps, cfg, bs, dat_type, stop_t_w2v = True):
    if cfg.LSTM.stop_t == False:
        stop_token = [[],[],[],[],[],[],[],[],[],[]]
    else:
        stop_token = [[cfg.LSTM.stop_t],[2],[2],[7],[2],[859],[0],[0],[34278],[cfg.LSTM.cut_off*852-1]]
    
    paths = np.array([[],[],[],[],[],[],[],[],[],[]])
    for i, pep in enumerate(peps):
        try:
            data = np.load(Path(cfg.DATA.PATH_TO_DATA_DIR) / f"Data/{pep}/prepared_data_{dat_type}_relabeled.npy")

            data_path = data[0,:].copy()
            data_path[data_path>=cfg.LSTM.cut_off]=cfg.LSTM.cut_off
            data_path +=cfg.LSTM.cut_off*pep

        
            data = np.insert(data, len(data), data_path, axis=0)
            paths = np.concatenate((paths,stop_token,data),axis=1)
        except FileNotFoundError:
            print(f"File {pep} not found")
            continue
    
    if paths.shape[1]%bs != 0:
        if dat_type == "test":
            paths = paths[:,:-(paths.shape[1]%bs)]
        else:
            paths = paths[:,(paths.shape[1]%bs):]
    paths = paths.reshape(len(stop_token), bs, -1)
    paths = torch.tensor(paths).permute(0,2,1)
    return paths

def load_sensible_peps(cfg):
    pep_info = pd.read_pickle(Path(cfg.DATA.PATH_TO_DATA_DIR) / "data_inf.pkl")
    pep_long = list(pep_info[(pep_info["length"]>cfg.LSTM.seq_len)].index)
    good_peps = np.loadtxt(Path(cfg.DATA.PATH_TO_DATA_DIR) / "good_peps.txt")
    peps = np.array(list(set(good_peps).intersection(set(pep_long)))).astype(int)
    # split list in train, test and validation
    peps_train = peps[:int(cfg.LSTM.train_percentage*len(peps))]
    peps_val = peps[int(cfg.LSTM.train_percentage*len(peps)):]
    return peps, peps_train, peps_val


def load_sensible_train_test_set(peps, cfg):
    dat_train = load_sensible(peps, cfg, cfg.TRAIN.BATCH_SIZE, "train")
    
    dat_test = {}
    for pep in peps:
        dat_t = load_sensible([pep], cfg, 1, "test")
        dat_w = load_sensible([pep], cfg, 1, "train")
        dat_test[pep] = {"warm": dat_w, "test": dat_t}
    
    return dat_train, dat_test

def load_locations(peps,cfg):
    locations = np.zeros((859+1,100,2))
    for i,pep in enumerate(peps):
        try:
            data = np.load(Path(cfg.DATA.PATH_TO_DATA_DIR) / f"Data/{pep}/prepared_data_locations_relabeled.npy")
            locations[pep][:data.shape[0],:data.shape[1]] = data
        except FileNotFoundError:
            print(f"File {pep} not found")
            continue   

    # Reshapes the list into a matrix with the batch size. 
    locations = torch.tensor(locations)[:,:cfg.LSTM.cut_off,:]
    return locations
