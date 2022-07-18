import torch
import numpy as np
from functools import partial
from collections import ChainMap, OrderedDict
from pytorch_lightning.loops.epoch import EvaluationEpochLoop
from pytorch_lightning.loops import EvaluationLoop
import torch.nn.functional as F
from typing import Any
from deprecate.utils import void

from nextlocationprediction.datasets.data import Sensible_dataset

def compute_logits(model, inp, target, bs, out, data_loc, cfg):
    tmp = target.clone()                                # Cloning target tensor to construct logits
    tmp[0][target[0]>cfg.LSTM.cut_off]=cfg.LSTM.cut_off                            # Setting all location bigger than 20 to 20
    logits = torch.Tensor(inp.shape[1], bs,cfg.LSTM.cut_off)   # Initialize logits tensor
    
    # Create other possible targets for every given location to compute logits for them
    for i in range(cfg.LSTM.cut_off):
        tmp[-1] = (tmp[-1]+1)%cfg.LSTM.cut_off+(tmp[-1]//cfg.LSTM.cut_off)*cfg.LSTM.cut_off  # Add 1 to the type data and keeps it in the same same interval
        tmp_20 = tmp[0]==cfg.LSTM.cut_off                        # Remembering which locations are already 20 for the rank data 
        tmp[0] = (tmp[0]+1)%cfg.LSTM.cut_off                     # Add 1 to the rank data mod 20
        tmp[0][tmp_20] = cfg.LSTM.cut_off                        # Setting those who were already 20 to 20 again
        logits[:,:,i] = torch.sum((out*model.embeddings_out(tmp.clone())),dim=2)  # Compute the logits of the output
        
    logits_dist = torch.Tensor(inp.shape[1],bs,cfg.LSTM.cut_off)   # Initialize logits tensor
    dat_loc_perm = data_loc[inp[5].long()].permute(3,0,1,2)
    inp_perm = inp[-4:-2].unsqueeze(3).expand_as(dat_loc_perm)
    logits_dist = torch.norm(inp_perm-dat_loc_perm,dim=0)*6371*np.pi/180  
    
    logits_dist[logits_dist>100]=100
    logits_dist/=100

    logits_dist = lognorm(logits_dist)
    logits_dist[logits_dist != logits_dist]=0  # Setting NaN values to zero
    
    logits = logits + logits_dist.float()*model.weight_dist

    
    # Computing the softmax of the logits
    # Adding a zero columns to predictions to make it the same size as target and fake "explorations" which can never be predicted.
    sm = F.pad(torch.softmax(logits,-1), (0,1))
    # Set the probability of the self location to zero
    for i in range(sm.shape[0]):
        for j in range(sm.shape[1]):
            if inp[0,i,j]<cfg.LSTM.cut_off:
                sm[i,j,int(inp[0,i,j])] = 0

    #sm_expl = torch.softmax(logits_expl,-1)
    
    target[0][target[0]>cfg.LSTM.cut_off] = cfg.LSTM.cut_off # Changing all target locations above 20 to 20.

    return logits.reshape(-1,cfg.LSTM.cut_off), target, sm

def lognorm(data, shape=2.292614965084133):
        return 1/(shape*data*np.sqrt(2*np.pi))*torch.exp(-torch.log(data)**2/(2*shape**2))

class LSTM_validation_loop(EvaluationLoop):
    def __init__(self):
        super().__init__()

    def run(self, *args: Any, **kwargs: Any):
        """The main entry point to the loop.

        Will frequently check the :attr:`done` condition and calls :attr:`advance`
        until :attr:`done` evaluates to ``True``.

        Override this if you wish to change the default behavior. The default implementation is:

        Example::

            def run(self, *args, **kwargs):
                if self.skip:
                    return self.on_skip()

                self.reset()
                self.on_run_start(*args, **kwargs)

                while not self.done:
                    self.advance(*args, **kwargs)

                output = self.on_run_end()
                return output

        Returns:
            The output of :attr:`on_run_end` (often outputs collected from each step of the loop)
        """
        if self.skip:
            return self.on_skip()

        self.reset()

        self.on_run_start(*args, **kwargs)

        # while not self.done:
        #     try:
        count = 0
        for pep in self.dataloaders[0]:
            self.trainer.lightning_module.net.hidden = self.trainer.lightning_module.net.init_hidden(bs = 1)
            for (inp, _, _, _ )in self.dataloaders[0].dataset.val_warm_dat(int(pep), "warm"):
                _, self.trainer.lightning_module.net.hidden, _ = self.trainer.lightning_module.net(inp)
            dat_load_pep = self.dataloaders[0].dataset.val_warm_dat(int(pep), "val")
            for _ in range(len(dat_load_pep)):
                self.on_advance_start(dat_load_pep, *args, **kwargs)
                self.advance(dat_load_pep, 0, *args, **kwargs)
                self.on_advance_end()
                self._restarting = False
                count += 1
            # except StopIteration:
            #     break
        self._restarting = False

        output = self.on_run_end()
        return output

    def advance(self, data_loader, idx, *args: any, **kwargs: any):
            """performs evaluation on one single dataloader."""
            void(*args, **kwargs)

            dataloader_idx = idx
            dataloader = data_loader 
            assert self._data_fetcher is not None
            self._data_fetcher.setup(
                dataloader,
                batch_to_device=partial(self.trainer._call_strategy_hook, "batch_to_device", dataloader_idx=dataloader_idx),
            )
            dl_max_batches = self._max_batches[dataloader_idx]

            kwargs = OrderedDict()
            if self.num_dataloaders > 1:
                kwargs["dataloader_idx"] = dataloader_idx
            dl_outputs = self.epoch_loop.run(self._data_fetcher, dl_max_batches, kwargs)

            # store batch level output per dataloader
            self._outputs.append(dl_outputs)

            if not self.trainer.sanity_checking:
                # indicate the loop has run
                self._has_run = True
    
    def on_advance_start(self, dataloader, *args: Any, **kwargs: any) -> None:
        # dataloader = self.current_dataloader
        if (
            dataloader is not None
            and getattr(dataloader, "sampler", None)
            and callable(getattr(dataloader.sampler, "set_epoch", None))
        ):
            # set seed for distributed sampler (enables shuffling for each epoch)
            dataloader.sampler.set_epoch(self.trainer.fit_loop.epoch_progress.current.processed)

        # super().on_advance_start(*args, **kwargs)



