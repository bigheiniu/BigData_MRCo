from Trainer.simple_trainer import SimpleTrainer
from Model.BasicEncoder import CNN_Text, RoBERTa_Text
from Model.MetaWeight import FullWeightModel
from Trainer.k_step_meta import step_hmlc_K
from Trainer.meta_process import step_l2w_group_net, step_l2w_group_net_previous
import torch
import numpy as np
from argparse import ArgumentParser
from transformers import get_scheduler
from Dataset.dataset import OfflineAugmentDatasetSpeedUp, collate_fn
from torch.utils.data import Subset
from torch import nn
from Model.Scheduler import ReduceLROnPlateau
# Dataset:
# train_clean, train_augment, train_clean_eval.

class DummyScheduler(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])

        return lrs

    def step(self, epoch=None):
        pass

class MetaTrainer(SimpleTrainer):
    def __init__(self, hparams):
        super(MetaTrainer, self).__init__(hparams)
        print(hparams)
        self.is_roberta = getattr(hparams, 'is_roberta', False)
        if self.is_roberta is False:
            self.basic_model = CNN_Text(hparams)
        else:
            self.basic_model = RoBERTa_Text(hparams)
        self.meta_weight = FullWeightModel(hparams)
        self.dw_prev = [0 for p in self.meta_weight.parameters() if p.requires_grad]
        self.automatic_optimization = False
        self.augment_dropout = nn.Dropout(p=getattr(hparams, "aug_dropout", 0))
        if getattr(self.hparams, "is_scheduler_noise", False):
            self.drop_out_scheduler = ReduceLROnPlateau("dropout",
                                                    factor=0.9,
                                                    eps=0.01,
                                                    patience=1,
                                                    threshold=getattr(self.hparams, "scheduler_threshold", 1e-2)
                                                    )

    def pre_step(self, x_clean, y_clean,  basic_optimizer, meta_optimizer):
        # first pre-train the base encoder, then pre-train the meta model.
        # pre-train the base encoder
        loss_basic = None
        loss_meta = None

        if self.current_epoch < self.hparams.pre_train_basic_epochs:
            loss_basic = self.basic_model.forward(x_clean, y_clean, is_reduction=True)[-1]
            basic_optimizer.zero_grad()
            self.manual_backward(loss_basic, basic_optimizer)
            basic_optimizer.step()

        # pre-train the meta model
        elif self.current_epoch - self.hparams.pre_train_basic_epochs < self.hparams.pre_train_meta_epochs:
            output = self.basic_model.forward(x_clean, y_clean, is_reduction=True)
            x_feature = output[0]
            loss_meta = self.meta_weight.pre_train(x_feature=x_feature, y_label=y_clean)
            meta_optimizer.zero_grad()
            self.manual_backward(loss_meta, meta_optimizer)
            meta_optimizer.step()

        if loss_meta is not None:
            return {"loss_meta": loss_meta}
        elif loss_basic is not None:
            return {"loss_basic": loss_basic}
        else:
            raise NotImplementedError("Current Epoch {} is greater than {}+{}".format(self.current_epoch,
                                                                                      self.hparams.pre_train_basic_epochs,
                                                                                      self.hparams.pre_train_meta_epochs
                                                                                      ))

    def validation_step(self, *args, **kwargs):
        batch = args[0]
        x_clean, y_clean = batch
        _, logits, loss = self.basic_model(x_clean, y_clean)
        return {"logits":logits, "loss":loss, "target":y_clean}

    def training_step(self, batch, batch_idx, optimizer_idx):
        basic_batch = batch['base_loader']
        meta_batch = batch['meta_loader']
        x_clean, y_clean, x_aug, y_aug = basic_batch
        # flat the dataset
        x_aug = x_aug.reshape(-1, x_aug.shape[-1])
        y_aug = y_aug.reshape(-1,)
        x_clean_val, y_clean_val = meta_batch
        (basic_optimizer, meta_optimizer) = self.optimizers()
        if self.is_roberta:
            (basic_scheduler, meta_scheduler) = self.lr_schedulers()
        # pre-train the meta model and basic encoder iteratively

        if self.current_epoch < (self.hparams.pre_train_basic_epochs + self.hparams.pre_train_meta_epochs)\
            and self.meta_weight.weight_output_dim == 1:
            # the pre-training is only for instance level weight
            loss = self.pre_step(x_clean, y_clean, basic_optimizer, meta_optimizer)
            return {"loss_pre": loss}
        else:
            # 1. My version 2. Guoqing version 3. Guoqing aaai
            meta_fn = step_hmlc_K if self.hparams.is_kstep_meta else step_l2w_group_net
            loss_val, loss_s, log_loss_s, loss_train_clean, loss_final, instance_weight = meta_fn(self,
                               main_net=self.basic_model, main_opt=basic_optimizer,
                               meta_net=self.meta_weight, meta_opt=meta_optimizer,
                               clean_train_input={"x": x_clean, "y": y_clean},
                               clean_val_input={"x": x_clean_val, "y": y_clean_val},
                               train_aug_data={"x": x_aug, 'y': y_aug})

            tensorboard_logs = {"loss":loss_final, "loss_s":loss_s, "mean_loss_s":log_loss_s, 'loss_train_clean':loss_train_clean, 'loss_val':loss_val}
        if self.is_roberta:
            basic_scheduler.step()
        return {"loss": loss_final, "log":tensorboard_logs, "instance_weight":instance_weight}


    def training_epoch_end(self, outputs):
        # outputs = outputs[0]
        if "loss_pre" in outputs[0]:
            loss_basic = [i['loss_pre']['loss_basic'].item() for i in outputs if 'loss_basic' in i['loss_pre']]
            loss_meta = [i['loss_pre']['loss_meta'].item() for i in outputs if 'loss_meta' in i['loss_pre']]
            if len(loss_basic) > 0:
                loss_basic = np.mean(loss_basic)
                self.log("Train/" + "Pre_Train_Loss_Basic",
                                              loss_basic, sync_dist=True)
            if len(loss_meta) > 0:
                loss_meta = np.mean(loss_meta)
                self.log("Train/" + "Pre_Train_Loss_Meta",
                                                  loss_meta, sync_dist=True)
        else:
            if type(outputs[0]) is list:
                outputs = outputs[0]
            loss = np.mean([i['loss'].item() for i in outputs])
            loss_s = np.mean([i['log']['loss_s'].item() for i in outputs])
            mean_loss_s = np.mean([i['log']['mean_loss_s'].item() for i in outputs])
            loss_train_clean = np.mean([i['log']['loss_train_clean'].item() for i in outputs])
            loss_g = np.mean([i['log']['loss_val'].item() for i in outputs])


            self.log("Train/" + "Loss",
                                              loss, sync_dist=True)
            self.log("Train/" + "Loss_s",
                                              loss_s, sync_dist=True)
            self.log("Train/" + "Mean_loss_s",
                                              mean_loss_s, sync_dist=True)
            self.log("Train/" + "loss_val",
                                              loss_g, sync_dist=True)
            self.log("Train/" + "Loss_train_clean",
                                              loss_train_clean, sync_dist=True)
            if "instance_weight" in outputs[0]:
                try:
                    instance_weight = torch.cat([i['instance_weight'].detach() for i in outputs], dim=0)
                    # update the dropout rate of the augmented samples.
                    instWeightNorm2 = instance_weight.norm(2)
                    if getattr(self.hparams, "is_scheduler_noise", False):
                        # increase the dropout if the module did not learn much knowledge
                        new_p = self.drop_out_scheduler.step(hyper_value=(1 - self.augment_dropout.p),
                                                 metrics=instWeightNorm2,
                                                 epoch=self.current_epoch)
                        self.augment_dropout.p = 1 - new_p
                        self.log("Train/AugDropOut", 1-new_p, sync_dist=True)
                    self.log("Train/InstWeightNorm2", instWeightNorm2, sync_dist=True)
                    # instance_weight = np.concatenate([i['instance_weight'].detach().cpu().numpy() for i in outputs], axis=0)
                    # instance_weight_np = instance_weight.cpu().numpy()
                    # self.logger.experiment.add_histogram("Train/InstanceWeight", instance_weight_np, self.current_epoch, sync_dist=True)
                except:
                    pass

    def train_dataloader(self):
        # multiple dataloader and a special one for validation
        train_loader = self.get_loader("train")
        train_meta_loader = self.get_loader("train", is_meta=True)
        return {"base_loader": train_loader, "meta_loader": train_meta_loader}

    def get_loader(self, train_type, is_meta=False):
        dataset = OfflineAugmentDatasetSpeedUp(self.hparams, train_type, is_meta=is_meta)
        if train_type == "train":
            shuffle = True
            batch_size = self.hparams.train_batch_size
        else:
            shuffle = False
            batch_size = self.hparams.eval_batch_size
        # without drop_last will cause the spikes in the training loss
        # pls check https://stacko  verflow.com/questions/47824598/why-does-my-training-loss-have-regular-spikes
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle)
        return dataloader

    def configure_optimizers(self):
        # TODO: check the scheduler
        if self.hparams.is_kstep_meta:
            if self.hparams.is_roberta:
                Optimizer_basic = torch.optim.Adam
            else:
                Optimizer_basic = torch.optim.SGD
        else:
            Optimizer_basic = torch.optim.Adam
        basic_optimizer = Optimizer_basic(
            filter(lambda p: p.requires_grad, self.basic_model.parameters()),
            lr=self.hparams.basic_lr_rate,
            weight_decay=self.hparams.basic_weight_decay
        )
        meta_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.meta_weight.parameters()),
            lr=self.hparams.meta_lr_rate,
            weight_decay=self.hparams.meta_weight_decay
        )
        if self.hparams.is_roberta:
            # ATTENTION: Linear Scheduler but the optimizer is SGD not Adamw.
            basic_scheduler = get_scheduler(
                "linear",
                basic_optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams.epochs * len(self.train_dataloader()['base_loader'].dataset),
            )
            # meta module did not utilize the lr scheduler
            meta_scheduler = DummyScheduler(meta_optimizer)
            return [basic_optimizer, meta_optimizer], [basic_scheduler, meta_scheduler]
        else:
            return [basic_optimizer, meta_optimizer]


        #TODO: Add scheduler in the training

        # basic_scheduler = {"scheduler":torch.optim.lr_scheduler.ReduceLROnPlateau(basic_optimizer, "min"),
        #                    'name': 'learning_rate_basic',
        #                    'interval': 'step',
        #                    'frequency': 1}
        #
        # meta_scheduler = {"scheduler":torch.optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, "min"),
        #                   'name': 'learning_rate_meta',
        #                   'interval': 'step',
        #                   'frequency': 1}
        #  [basic_scheduler, meta_scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--meta_lr_rate", default=0.01, type=float)
        parser.add_argument("--meta_weight_decay", default=0.001, type=float)
        parser.add_argument("--cls_emb_dim", default=128, type=int)
        parser.add_argument("--gw_hidden_dim", default=256, type=int)
        parser.add_argument("--is_deeper_weight", action='store_true')
        parser.add_argument("--gw_dropout", default=0.3, type=float)
        parser.add_argument("--gradient_steps", default=1, type=int, help='Number of look-ahead gradient steps for meta-gradient (default: 1)')
        parser.add_argument("--is_kstep_meta", action='store_true', help='whether utilize Guoqing s method in aaai')
        parser.add_argument("--is_debug_label_flip", action='store_true', help='debug the meta-weight model by adding label noise into the dataset. ')
        parser.add_argument("--debug_flip_pro", default=0.2, help="label flip probability")
        parser.add_argument("--pre_train_basic_epochs", default=0, type=int)
        parser.add_argument("--pre_train_meta_epochs", default=0, type=int)
        parser.add_argument("--num_cola_score_histograms", default=100, type=int)
        parser.add_argument("--num_read_score_histograms", default=100, type=int)
        parser.add_argument("--histogram_embed_size", default=-1, type=int, help="-1 means use the raw cola/read score as the feature for meta-weight model")
        parser.add_argument("--is_score_embed", action='store_true',help='whether utilize the auxiliary score like readability/cola for the meta-weight model')
        parser.add_argument("--is_pre_meta_metric", action='store_true',help='whether use the l1 distance for pre-training')
        parser.add_argument("--weight_output_dim", default=1, type=int,
                            help='if > 1 will apply the logits weight， '
                                 'this should match the number of class')
        parser.add_argument("--weight_scale", default=1.0, type=float,
                            help='')
        parser.add_argument("--is_add_noise", action="store_true")
        parser.add_argument("--is_scheduler_noise", action="store_true")
        parser.add_argument("--aug_dropout", default=0.2, type=float)
        parser.add_argument("--scheduler_threshold", default=1e-2, type=float)
        return parser


class MetaTrainerLast(MetaTrainer):
    def __init__(self, hparams):
        super(MetaTrainer, self).__init__(hparams)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_clean, y_clean, x_clean_val, y_clean_val, x_aug, y_aug, _, _ = batch
        basic_optimizer, _ = self.optimizers()
        x = torch.cat((x_clean, x_clean_val), dim=0)
        y = torch.cat((y_clean, y_clean_val), dim=0)
        _, logits, loss = self.basic_model(x, y)
        basic_optimizer.zero_grad()
        self.manual_backward(loss, basic_optimizer)
        basic_optimizer.step()
        tensorboard_logs = {"loss": loss, "loss_s": loss, 'loss_train_clean': loss,
                            'loss_val': loss}
        return {"loss": loss, "log": tensorboard_logs}
