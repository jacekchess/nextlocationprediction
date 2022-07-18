import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .utils.parser import load_config, parse_args
from .models.build import build_model
from .datasets.build import build_dataset 
from .models.utils import LSTM_validation_loop, compute_logits
import nextlocationprediction.models.optimizer as optim
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import nextlocationprediction.models.models # Needed to see models
import nextlocationprediction.datasets.data # Needed to see models
from torchmetrics import Accuracy


class LightningModel(pl.LightningModule):
    def __init__(self, net, cfg):
        super().__init__()
        self.net = net
        self.cfg = cfg
        self.acc_1_train = Accuracy(top_k=1)
        self.acc_5_train = Accuracy(top_k=5)
        self.acc_1_val = Accuracy(top_k=1)
        self.acc_5_val = Accuracy(top_k=5)

    def forward(self, x):
        return self.net(x)

    ### TODO
    def training_step(self, batch, batch_idx):
        inp, target, target_expl, data_loc = [torch.squeeze(x, dim=0) for x in batch]
        out, _, logits_expl = self.net(inp)

        logits, targets_cut, sm = compute_logits(self.net, inp, target, self.cfg.TRAIN.BATCH_SIZE, out, data_loc, self.cfg)

        # train_acc = torch.sum(sm.argmax(dim=2) == target_cut[0])
        # Adding a zero columns to predictions to make it the same size as target and fake "explorations" which can never be predicted.
        train_acc_1 = self.acc_1_train(sm.flatten(end_dim=1), targets_cut[0].flatten())
        train_acc_5 = self.acc_5_train(sm.flatten(end_dim=1), targets_cut[0].flatten())
        loss = F.cross_entropy(logits, targets_cut[0].reshape(-1), ignore_index = self.cfg.LSTM.cut_off)

        self.log("train_loss", loss, on_step=True, on_epoch=False, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log("train_top1", train_acc_1,  on_step=False, on_epoch=True, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log("train_top5", train_acc_5,  on_step=False, on_epoch=True, batch_size=self.cfg.TRAIN.BATCH_SIZE)
    
        return {"loss": loss}

    ### TODO
    def validation_step(self, batch, batch_idx):
        inp, target, target_expl, data_loc = [torch.squeeze(x, dim=0) for x in batch]
        out, _, logits_expl = self.net(inp)

        logits, targets_cut, sm = compute_logits(self.net, inp, target, 1, out, self.trainer.val_dataloaders[0].dataset.data_loc, self.cfg)

        # val_acc = torch.sum(sm.argmax(dim=2) == targets_cut[0])
        val_acc_1 = self.acc_1_val(sm.flatten(end_dim=1), targets_cut[0].flatten())
        val_acc_5 = self.acc_5_val(sm.flatten(end_dim=1), targets_cut[0].flatten())
        loss = F.cross_entropy(logits, targets_cut[0].reshape(-1), ignore_index = self.cfg.LSTM.cut_off)

        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=1)
        self.log("val_top1", val_acc_1, on_step=False, on_epoch=True, batch_size=1)
        self.log("val_top5", val_acc_5, on_step=False, on_epoch=True, batch_size=1)
        return {"val_loss": loss, "val_top1": val_acc_1}

    ### TODO
    def validation_epoch_end(self, outputs):
        # avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        # Reset hidden
        self.trainer.lightning_module.net.hidden = self.trainer.lightning_module.net.init_hidden(self.cfg.TRAIN.BATCH_SIZE)
        return None
        # return {"Epoch_loss": avg_loss}

    ### TODO
    def configure_optimizers(self):
        optimizer = optim.construct_optimizer(self.net, self.cfg)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.SOLVER.LR_MILESTONES, gamma=self.cfg.SOLVER.LR_GAMMA)
        return [optimizer], [lr_scheduler]
    # def on_before_backward(self, loss: torch.Tensor) -> None: 
    #     self.net.repackage_hidden()

def main():
    args = parse_args()
    cfg = load_config(args)
    wandb_logger = WandbLogger(project="nextlocationprediction", name=cfg.MODEL.MODEL_NAME, save_dir=cfg["OUTPUT_DIR"])

    data_module = build_dataset(cfg)
    net = build_model(cfg)
    # print(net)

    pl_module = LightningModel(net, cfg)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.OUTPUT_DIR, save_top_k=2, monitor="val_loss")

    trainer = pl.Trainer(
        gpus=cfg.NUM_GPUS,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        enable_model_summary=False,
        enable_progress_bar=False,
        max_epochs=cfg.TRAIN.MAX_EPOCHS,
        callbacks=[lr_monitor, checkpoint_callback],
        default_root_dir=cfg.OUTPUT_DIR,
        )
    # trainer.validate_loop = LSTM_validation_loop()

    ### Changing the loop so the validation is done on the validation set with the double loop
    # Optional: stitch back the trainer arguments
    epoch_loop = trainer.fit_loop.epoch_loop 
    # Optional: connect children loops as they might have existing state
    epoch_loop.connect(trainer.fit_loop.epoch_loop.batch_loop, LSTM_validation_loop())
    # Instantiate and connect the loop.
    trainer.fit_loop.connect(epoch_loop=epoch_loop)
    trainer.fit(pl_module, data_module)

if __name__ == "__main__":
    main()