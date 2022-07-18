import pytorch_lightning
import numpy as np
import torch


import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from .build import DATASET_REGISTRY
from .utils import load_sensible_train_test_set, load_sensible_peps, load_locations

torch.manual_seed(0)

@DATASET_REGISTRY.register()
class Sensible(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.peps, self.peps_train, self.peps_val = load_sensible_peps(self.cfg)
        _, self.dat_test = load_sensible_train_test_set(self.peps_train, self.cfg)
        self.dat_train_both, self.dat_test_all = load_sensible_train_test_set(self.peps, self.cfg)
        _, self.dat_test_vali = load_sensible_train_test_set(self.peps_val, self.cfg)
        self.data_loc = load_locations(self.peps, self.cfg)

    def train_dataloader(self):
        return DataLoader(
            Sensible_dataset(self.cfg, self.dat_train_both, "train", self.data_loc), 
            num_workers=self.cfg.DATA_LOADER.NUM_WORKERS,
            shuffle=False, 
            pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY
            )

    def val_dataloader(self):
        return DataLoader(
            Sensible_dataset(self.cfg, self.dat_test, "val", self.data_loc), 
            num_workers=self.cfg.DATA_LOADER.NUM_WORKERS,
            shuffle=False, 
            pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY
            )

    
    def configure_optimizers(self):
        optimizer = optim.construct_optimizer(self.net, self.cfg)
        # Faking the LR_schedular used in TimeSformer which is a stepwise LR depending on the epoch.
        lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.SOLVER.STEPS, gamma=0.1)
        return [optimizer], [lr_schedular]


class Sensible_dataset(Dataset):

    def __init__(self, cfg, data, dat_type, data_loc):
        """
        Args:
            cfg: config file
            data: dataframe
        """
        self.cfg = cfg
        self.data = data
        self.dat_type = dat_type
        self.data_loc = data_loc

        if self.dat_type == "val_warm":
            self.rand_int = self.data.shape[1]%self.cfg.LSTM.seq_len  
        elif self.dat_type == "val_val":
            self.rand_int = 0
        elif self.dat_type == "val":
            return None
        elif self.dat_type == "train":
            self.rand_int = np.random.randint(self.cfg.LSTM.seq_len)  # Generates random number between 0 and seq_len.
        else:
            raise ValueError(f"dat_type not recognized: {dat_type}")
        self.ii_starts = np.arange(self.rand_int, self.data.shape[1]-2, self.cfg.LSTM.seq_len)  # Generates the array of start indices for each batch

    def __len__(self):
        if self.dat_type in  ["train", "val_val", "val_warm"]:
            return len(self.ii_starts)
        elif self.dat_type == "val":
            return len(self.data.keys())

    def __getitem__(self, idx):
        if self.dat_type in ["train", "val_val", "val_warm"]:
            # Generating start index depending on data type
            idx = self.ii_starts[idx] 
            seq = self.data[:,idx: idx+ self.cfg.LSTM.seq_len+1, :]
            inp = seq[:,:-1]
            target = seq[[0,5,8,-1],1:].long()
            target_expl = seq[[4],1:].long()
            return inp, target, target_expl, self.data_loc
        elif self.dat_type == "val":
            return list(self.data.keys())[idx] 
            # return [Sensible_dataset(self.cfg, self.data[idx]["warm"], "val_warm", self.data_loc), Sensible_dataset(self.cfg, self.data[idx]["test"], "val_val", self.data_loc)]
    
    def val_warm_dat(self, idx, dat_type):
        if dat_type == "warm":
            return Sensible_dataset(self.cfg, self.data[idx]["warm"], "val_warm", self.data_loc)
        if dat_type == "val":
            return Sensible_dataset(self.cfg, self.data[idx]["test"], "val_val", self.data_loc)


