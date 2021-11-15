from Trainer.simple_trainer import SimpleTrainer
from Model.BasicEncoder import CNN_Text, RoBERTa_Text
import torch
import numpy as np
from argparse import ArgumentParser
import torch.nn as nn
from transformers import AdamW, get_scheduler
from Util.util import evaluation
class BaselineTrainer(SimpleTrainer):
    def __init__(self, hparams):
        super(BaselineTrainer, self).__init__(hparams)
        if getattr(hparams, 'is_roberta', False) is False:
            self.basic_model = CNN_Text(hparams)
        else:
            self.basic_model = RoBERTa_Text(hparams)
            self.automatic_optimization = False

    def validation_step(self, *args, **kwargs):
        batch = args[0]
        x_clean, y_clean = batch
        _, logits, loss = self.basic_model(x_clean, y_clean)
        return {"logits": logits, "loss": loss, "target": y_clean}

    def training_step(self, *args, **kwargs):
        batch = args[0]
        x_clean, y_clean, _, _, _, _, x_aug, y_aug, _, _ = batch
        if self.hparams.no_aug_data:
            x = x_clean
            y = y_clean
        else:
            x = torch.cat((x_clean, x_aug), 0)
            y = torch.cat((y_clean, y_aug), 0)
        _, logits, loss = self.basic_model(x, y)
        tensorboard_logs = {"loss": loss}
        if self.hparams.is_roberta:
            # manually do the backpropagation for roberta model
            lr_sch = self.lr_schedulers()
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(loss, optimizer)
            optimizer.step()
            lr_sch.step()
            tensorboard_logs['lr'] = optimizer.defaults.get("lr", 0)

        return {"loss": loss, "log": tensorboard_logs, "logits":logits, 'target':y}

    def training_epoch_end(self, outputs):
        loss = np.mean([i['loss'].item() for i in outputs])
        logits = np.concatenate([x["logits"].detach().cpu().numpy() for x in outputs], axis=0)
        out_label_ids = np.concatenate([x["target"].cpu().numpy() for x in outputs], axis=0)
        result = evaluation(logits, out_label_ids, self.metric)
        self.logger.experiment.add_scalar("Train/" + "Loss",
                                          loss, self.current_epoch)
        for key, value in result.items():
            self.logger.experiment.add_scalar("Train/Epoch-" + key,
                                              value, self.current_epoch)

    @staticmethod
    def add_model_specific_args(parent_parser):
        #TODO: Other Hyper-parameter in the baseline methods? The augmented data selection?
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser