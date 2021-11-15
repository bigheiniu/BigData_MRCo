from Trainer.simple_trainer import SimpleTrainer
from Model.BasicEncoder import CNN_Text, RoBERTa_Text
from Dataset.dataset import OfflineAugmentDataset
from Trainer.k_step_meta import step_hmlc_K
from Trainer.meta_process import step_l2w_group_net, step_l2w_group_net_previous
import torch
import numpy as np
from argparse import ArgumentParser
from Trainer.meta_trainer import MetaTrainer
import torch.nn as nn
from copy import deepcopy
from math import ceil
import time
import torch.distributed as dist


# Dataset:
# train_clean, train_augment, train_clean_eval.
class AlignedMetaContrastTrainer(MetaTrainer):
    def __init__(self, hparams, is_half=False):
        super(AlignedMetaContrastTrainer, self).__init__(hparams)
        print(hparams)
        # copy the parameter from the basic_encoder
        if self.is_roberta is False:
            self.key_encoder = CNN_Text(hparams)
        else:
            self.key_encoder = RoBERTa_Text(hparams)
        self.is_ddp = getattr(hparams,"is_ddp", False)
        self.start_contrast = False
        self.is_half = is_half
        # 50 steps
        self.ttu = getattr(hparams, 'ttu', 50.0)
        self.float_dtype = torch.float16 if is_half else torch.float32
        self.register_buffer("queue", torch.randn(hparams.class_num, hparams.hidden_size, hparams.num_neg, dtype=self.float_dtype))
        self.register_buffer("queue_score", torch.ones(hparams.class_num, hparams.num_neg, dtype=self.float_dtype))
        self.register_buffer("queue_life_time",
                             torch.ones(hparams.class_num, hparams.num_neg, dtype=self.float_dtype) * self.ttu)

        # self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.m = hparams.momentum
        self.T = hparams.temperature
        self.base_temperature = hparams.base_temperature
        self.class_num = hparams.class_num

    # utils
    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        if self.is_ddp:
            print(self.global_rank, "start gathering")
            tensors_gather = [torch.ones_like(tensor)
                              for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
            print(self.global_rank, "gather before concat")

            output = torch.cat(tensors_gather, dim=0)
        else:
            output = tensor
        return output

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.basic_model.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _time_priority_dequeue_and_enqueue(self, keys, weight_score, label_index):
        # FIFO -> Time Aware Least Recent Used (TLRU)
        # keys shape: batch_size, hidden_dim
        # weight_score shape: batch_size, 1
        # labels shape: batch_size, 1

        # gather tensors from all devices
        # important: gathering does not collect gradients!!!
        keys = self.concat_all_gather(keys)
        weight_score = self.concat_all_gather(weight_score)
        # label_index = self.concat_all_gather(label_index)

        weight_score_one_dim = weight_score.squeeze(1)
        ascending_weight_index = torch.argsort(weight_score_one_dim, dim=0)

        self.queue_life_time -= 1
        idle_queue_index = (self.queue_life_time[label_index] <= 0).nonzero().squeeze(1)
        live_queue_index = (self.queue_life_time[label_index] > 0).nonzero().squeeze(1)
        # dequeue and enqueue by the lifetime
        time_dequeue_length = 0
        if len(idle_queue_index) > 0:
            time_dequeue_length = min(len(idle_queue_index), len(ascending_weight_index))
            enqueue_index = ascending_weight_index[-1 * time_dequeue_length:]
            ascending_time_index = torch.argsort(self.queue_life_time[label_index, idle_queue_index])
            dequeue_index = idle_queue_index[ascending_time_index[:time_dequeue_length]]

            self.queue[label_index, :, dequeue_index] = keys[enqueue_index].T
            self.queue_life_time[label_index, dequeue_index] = self.queue_life_time.new_ones(1, len(
                dequeue_index)) * self.ttu
            self.queue_score[label_index, dequeue_index] = weight_score_one_dim[enqueue_index]

        # dequeue and enqueue by the weight score
        left_index = ascending_weight_index[: len(ascending_weight_index)-time_dequeue_length]
        if len(live_queue_index) > 0 and len(left_index) > 0:
            # dequeue elements with large score, at the bottom of ascending_queue
            descending_queue_score, descending_queue_indices = torch.sort(self.queue_score[label_index, live_queue_index],
                                                                        descending=True)
            # elementwise comparison
            replace_region = torch.gt(descending_queue_score.unsqueeze(1),
                                     weight_score_one_dim[left_index].unsqueeze(0)).nonzero().to(label_index.device)
            # update queue
            if len(replace_region) > 0:
                # replace the **large** dequeue with **small** enqueue
                weight_dequeue_index = torch.unique(replace_region[:, 0], sorted=True)
                weight_enqueue_index = torch.unique(replace_region[:, 1], sorted=True)
                length = min(len(weight_dequeue_index), len(weight_enqueue_index))
                # pop up element with large score
                weight_dequeue_index = weight_dequeue_index[:length]
                # insert element with small score
                weight_enqueue_index = weight_enqueue_index[:length]

                dequeue_index = live_queue_index[descending_queue_indices[weight_dequeue_index]]
                enqueue_index = left_index[weight_enqueue_index]
                self.queue[label_index, :, dequeue_index] = keys[enqueue_index].T
                self.queue_score[label_index, dequeue_index] = weight_score_one_dim[enqueue_index]
                self.queue_life_time[
                    label_index, dequeue_index] = self.queue_life_time.new_ones(
                    len(dequeue_index)) * self.ttu

    @torch.no_grad()
    def _weight_priority_dequeue_and_enqueue(self, keys, weight_score, label_index):
        # FIFO -> Time Aware Least Recent Used (TLRU)
        # keys shape: batch_size, hidden_dim
        # weight_score shape: batch_size, 1
        # labels shape: batch_size, 1
        print(self.global_rank, "start of dequeue enqueue")

        # gather tensors from all devices
        # important: gathering does not collect gradients!!!
        keys = self.concat_all_gather(keys)
        weight_score = self.concat_all_gather(weight_score)
        # label_index = self.concat_all_gather(label_index)
        print(self.global_rank, "end of gathering")

        weight_score_one_dim = weight_score.squeeze(1)
        ascending_weight_index = torch.argsort(weight_score_one_dim, dim=0)
        print(self.global_rank, "end of argsort")
        self.queue_life_time -= 1
        print(self.global_rank, "end of update queue life time")
        print(self.global_rank, self.queue_score.shape)
        print(self.queue_score[label_index,:].shape, max(self.queue_score[label_index,:]))
        # dequeue elements with large score, at the bottom of ascending_queue
        descending_queue_score, descending_queue_indices = torch.sort(
            self.queue_score[label_index, :],
            descending=True)
        print(self.global_rank, "end of sorting")
        # elementwise comparison
        replace_region = torch.gt(descending_queue_score.unsqueeze(1),
                                  weight_score_one_dim.unsqueeze(0)).nonzero().to(weight_score.device)
        weight_replace_length = 0
        print(self.global_rank, "start update queue")
        # update queue
        if len(replace_region) > 0:
            # replace the **large** dequeue with **small** enqueue
            weight_dequeue_index = torch.unique(replace_region[:, 0], sorted=True)
            weight_enqueue_index = torch.unique(replace_region[:, 1], sorted=True)
            weight_replace_length = min(len(weight_dequeue_index), len(weight_enqueue_index))
            # pop up element with large score
            weight_dequeue_index = weight_dequeue_index[:weight_replace_length]
            # insert element with small score
            weight_enqueue_index = weight_enqueue_index[:weight_replace_length]

            dequeue_index = descending_queue_indices[weight_dequeue_index]
            enqueue_index = ascending_weight_index[weight_enqueue_index]
            # self.queue_score[label_index, dequeue_index] = weight_score_one_dim[enqueue_index]
            self.queue[label_index, :, dequeue_index] = keys[enqueue_index].T
            self.queue_score[label_index, dequeue_index] = weight_score_one_dim[enqueue_index]
            self.queue_life_time[
                label_index, dequeue_index] = self.queue_life_time.new_ones(
                len(dequeue_index)) * self.ttu

        idle_queue_index = (self.queue_life_time[label_index] <= 0).nonzero().squeeze(1)
        print(self.global_rank, "start de en by lifetime")
        # dequeue and enqueue by the lifetime
        if len(idle_queue_index) > 0 and len(ascending_weight_index) > weight_replace_length:
            # remove the eldest elements
            ascend_idle_queue_index = torch.argsort(self.queue_life_time[label_index, idle_queue_index])
            enqueue_index = ascending_weight_index[weight_replace_length:]
            time_replace_length = min(len(enqueue_index), len(ascend_idle_queue_index))
            ascend_idle_queue_index = idle_queue_index[ascend_idle_queue_index[:time_replace_length]]
            enqueue_index = enqueue_index[:time_replace_length]

            self.queue[label_index, :, ascend_idle_queue_index] = keys[enqueue_index].T

            self.queue_life_time[label_index, ascend_idle_queue_index] = self.queue_life_time.new_ones(1, len(
                ascend_idle_queue_index)) * self.ttu

            self.queue_score[label_index, ascend_idle_queue_index] = weight_score_one_dim[enqueue_index]
        print(self.global_rank, "end of dequeue enqueue")

    def inner_contrast_learning(self, clean_data_feature, aug_data, aug_data_feature, weight_score, label_index, exclusive_flag=None,
                                normalize_vector=False, pos_ratio=1, queue_exclusive=False,
                                ):
        print(self.global_rank, "start of inner")
        # clean_data: N, seq_len
        # aug_data: N , seq_len
        # weight_score: N, 1
        q = clean_data_feature
        # q_aug
        q_aug = aug_data_feature
        # loss criterion
        loss_fn = torch.nn.CrossEntropyLoss()
        queue = self.queue.clone().detach()[label_index]

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            pos_k = self.key_encoder(aug_data)[0]
        # tuning hyper
        if normalize_vector:
            # Enqueue element have already been normalized
            # we were not using BN, so no need to gather here
            pos_k = nn.functional.normalize(pos_k, dim=-1)
            if self.is_half:
                pos_k = pos_k.half()

            # except:
            #     print("ERROR HERE!!!")
            #     print(label_index)

        # select samples for pos by ratio
        if len(q) > 1:
            chosen_size = max(int(pos_ratio * len(q)), 1)
            chosen_weights, chosen_indices = torch.sort(weight_score.squeeze(), descending=True)
            chosen_indices = chosen_indices[:chosen_size]
            chosen_weights = chosen_weights[:chosen_size]
            # remove samples with smaller weight
            if queue_exclusive:
                queue_score = self.queue_score[label_index].clone().detach().squeeze()
                # do not compare against init values
                actual_weight_indices = torch.ne(queue_score, 1).nonzero()
                if len(actual_weight_indices) > 0:
                    max_queue_score = queue_score[actual_weight_indices].max()
                    exclusive_indices = torch.gt(chosen_weights, max_queue_score).nonzero().squeeze()
                    chosen_indices = chosen_indices[exclusive_indices]
        
            if len(chosen_indices) == 0:
                # dummy loss, does it work if i just return 0?
                dummy_label = torch.zeros(1, dtype=torch.long, device=clean_data_feature.device)
                dummy_logits = torch.tensor([1.,0.]).reshape(1,-1).to(clean_data_feature.device)
                return loss_fn(dummy_logits, dummy_label)
        
            q = q[chosen_indices, :]
            q_aug = q_aug[chosen_indices, :]
            pos_k = pos_k[chosen_indices, :]

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_q = torch.einsum('nc,nc->n', q, pos_k).unsqueeze(-1)
        l_pos_aug = torch.einsum('nc,nc->n', q_aug, pos_k).unsqueeze(-1)
        ## low quality samples in the same class
        # negative logits: NxK
        # utilize the updated queue.
        l_neg_q = torch.einsum('nc,ck->nk', q, queue)
        # l_neg_aug = torch.einsum('nc,ck->nk', q_aug, self.queue.clone().detach()).unsqueeze(-1)
        l_neg_aug = torch.einsum('nc,ck->nk', q_aug, queue)

        l_pos = torch.cat([l_pos_q, l_pos_aug], dim=0)
        l_neg = torch.cat([l_neg_q, l_neg_aug], dim=0)
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = loss_fn(logits, labels)

        # dequeue and enqueue
        if self.hparams.is_time_prior:
            self._time_priority_dequeue_and_enqueue(pos_k, weight_score, label_index)
        else:
            self._weight_priority_dequeue_and_enqueue(pos_k, weight_score, label_index)
        print(self.global_rank, "end of inner")

        return loss

    def intra_contrast_learning(self, data_feature, labels):
        ## Modify "supervised contrastive learning"

        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # Approach 1: ignore the difference between the raw data and augmented data
        # Approach 2: omit the augmented data for intra contrastive learning, only utilize the original data
        device = data_feature.device
        batch_size = labels.shape[0]
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(data_feature, data_feature.T),
            self.T)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # remove self-contrast samples
        logits_mask = 1 - torch.eye(logits.shape[0], device=device)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = torch.mean(- (self.T / self.base_temperature) * mean_log_prob_pos)
        return loss

    def contrast_main(self, clean_data, aug_data, weight_score, labels, is_feature_normalized=False, pos_ratio=1, queue_exclusive=False):
        print(self.global_rank, "start of contrast")
        label_unique = torch.unique(labels)
        # compatible with logging function
        inner_contrast_loss = torch.tensor(0., device=weight_score.device)
        final_loss = torch.tensor(0., device=weight_score.device)
        intra_contrast_loss = torch.tensor(0., device=weight_score.device)

        # inner class contrastive learning
        clean_data_feature = self.basic_model(clean_data)[0]
        aug_data_feature = self.basic_model(aug_data)[0]
        assert self.hparams.is_weight_contrast + self.hparams.is_supervised_contrast > 0
        if is_feature_normalized:
            # instance based feature normalization
            clean_data_feature = nn.functional.normalize(clean_data_feature, dim=-1)
            aug_data_feature = nn.functional.normalize(aug_data_feature, dim=-1)
            if self.is_half:
                clean_data_feature = clean_data_feature.half()
                aug_data_feature = aug_data_feature.half()
        if self.hparams.is_weight_contrast:
            for label_index in range(self.class_num):
            # for label_index in label_unique:
                # label_exists = (labels == label_index).nonzero().shape[0] > 0
                # exclusive_flag = torch.tensor([1] if label_exists else [0],
                #                               device=labels.device)

                keep_index = (labels == label_index).nonzero().squeeze()
                # if len(keep_index.shape) == 1:
                #     keep_index=keep_index.unsqueeze(0)
                keep_clean_data_feature = clean_data_feature[keep_index]
                keep_aug_data_feature = aug_data_feature[keep_index]
                keep_weight_score = weight_score[keep_index]
                # len(keep_index), N, seq_len
                # flat
                keep_aug_data = aug_data[keep_index]

                if len(keep_aug_data.shape) == 1:
                    keep_aug_data = keep_aug_data.unsqueeze(0)
                    keep_aug_data_feature = keep_aug_data_feature.unsqueeze(0)
                    keep_weight_score = keep_weight_score.unsqueeze(0)
                if len(keep_clean_data_feature.shape) == 1:
                    keep_clean_data_feature = keep_clean_data_feature.unsqueeze(0)
                inner_contrast_loss += self.inner_contrast_learning(keep_clean_data_feature,
                                                                    keep_aug_data, keep_aug_data_feature, keep_weight_score,
                                                                    label_index, exclusive_flag=None, normalize_vector=is_feature_normalized, pos_ratio=pos_ratio, queue_exclusive=queue_exclusive)
                final_loss += self.hparams.lambda_inner_contrast * inner_contrast_loss
                # ATTENTION: set a barrier here
                # print("Hello")
                if self.is_ddp:
                    dist.barrier()
            # print("HHHHH")


        # TODO: apply to the augmented samples
        # only apply to samples inside the batch
        if self.hparams.is_supervised_contrast:
            # ATTENTION: this version only applied SupervisedContrast on clean samples
            if self.hparams.sc_include_aug:
                feature = torch.cat([clean_data_feature, aug_data_feature], dim=0)
                labels = labels.repeat(1 + int(aug_data_feature.shape[0] / clean_data_feature.shape[0]))
            else:
                feature = clean_data_feature
            intra_contrast_loss = self.intra_contrast_learning(feature, labels)
            final_loss += self.hparams.lambda_intra_contrast * intra_contrast_loss
        print(self.global_rank, "end of inner contrast")
        return final_loss, inner_contrast_loss, intra_contrast_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        print(self.global_rank, "start of step")
        basic_batch = batch['base_loader']
        meta_batch = batch['meta_loader']
        # x_aug shape: batch_size, N, seq_len
        x_clean, y_clean, x_aug, y_aug = basic_batch
        x_clean_val, y_clean_val = meta_batch
        (basic_optimizer, meta_optimizer) = self.optimizers()
        if self.is_roberta:
            (basic_scheduler, meta_scheduler) = self.lr_schedulers()
        # pre-train the meta model and basic encoder iteratively
        start_time = time.time()
        if self.current_epoch < (self.hparams.pre_train_basic_epochs + self.hparams.pre_train_meta_epochs) \
                and self.meta_weight.weight_output_dim == 1:
            # the pre-training is only for instance level weight

            loss_pre = self.pre_step(x_clean, y_clean, basic_optimizer, meta_optimizer)
            end_pre = time.time()
            self.time_pre += end_pre - start_time
            return {"loss_pre": loss_pre}
        else:
            meta_fn = step_hmlc_K if self.hparams.is_kstep_meta else step_l2w_group_net
            loss_val, loss_s, log_loss_s, loss_train_clean, loss_final, instance_weight = meta_fn(self,
                                                                                      main_net=self.basic_model,
                                                                                      main_opt=basic_optimizer,
                                                                                      meta_net=self.meta_weight,
                                                                                      meta_opt=meta_optimizer,
                                                                                      clean_train_input={"x": x_clean,
                                                                                                         "y": y_clean},
                                                                                      clean_val_input={"x": x_clean_val,
                                                                                                       "y": y_clean_val},
                                                                                      # flat the dataset
                                                                                      train_aug_data={"x": x_aug,
                                                                                                      'y': y_aug,})
            tensorboard_logs = {"loss": loss_final, "loss_s": loss_s, "mean_loss_s": log_loss_s,
                                'loss_train_clean': loss_train_clean, 'loss_val': loss_val}
            end_meta = time.time()
            self.time_meta += end_meta - start_time
            if self.hparams.contrast_epoch + self.hparams.contrast_patience \
                    >= self.current_epoch \
                    >= self.hparams.contrast_epoch:
                # TODO: alternate execution by epochs
                # since the data is aligned, y_clean == y_aug
                weight_score = instance_weight.detach()
                if self.hparams.is_weight_contrast + self.hparams.is_supervised_contrast > 0:
                    contrast_loss, inner_contrast_loss, intra_contrast_loss = self.contrast_main(
                        weight_score=weight_score,
                        clean_data=x_clean,
                        aug_data=x_aug,
                        labels=y_clean,
                        is_feature_normalized=self.hparams.is_feature_normalized,
                        pos_ratio=self.hparams.pos_ratio,
                        queue_exclusive=self.hparams.queue_exclusive)
                    basic_optimizer.zero_grad()
                    self.manual_backward(contrast_loss, basic_optimizer)
                    basic_optimizer.step()
                    tensorboard_logs.update({'inner_contrast_loss': inner_contrast_loss, 'intra_contrast_loss':intra_contrast_loss,
                                         "contrast_loss": contrast_loss})
                end_contrast = time.time()
                self.time_contrast += end_contrast - end_meta
            queue_score = self.queue_score.clone()
            queue_life_time = self.queue_life_time.clone()
        # ATTENTION: Change the Learning Rate when utilizing roberta model
        if self.is_roberta:
            basic_scheduler.step()
        print(self.global_rank, "end of step")
        return {"loss": loss_final, "log": tensorboard_logs, "instance_weight": instance_weight,
                "queue_score": queue_score, "queue_life_time": queue_life_time}

    def training_epoch_end(self, outputs):
        super().training_epoch_end(outputs)
        print(self.time_pre,self.time_meta, self.time_contrast)
        if type(outputs[0]) is list:
            outputs = outputs[0]
        if 'contrast_loss' in outputs[0].get('log', {}):
            # log loss
            inner_loss = np.mean([i['log']['inner_contrast_loss'].item() for i in outputs])
            intra_loss = np.mean([i['log']['intra_contrast_loss'].item() for i in outputs])
            contrast_loss = np.mean([i['log']['contrast_loss'].item() for i in outputs])

            self.log("Train/" + "Loss_c_inner",
                                            inner_loss, sync_dist=True)
            self.log("Train/" + "Loss_c_intra",
                                            intra_loss, sync_dist=True)
            self.log("Train/" + "Loss_c",
                                              contrast_loss, sync_dist=True)

    def on_train_epoch_start(self):
        self.time_pre = 0
        self.time_meta = 0
        self.time_contrast = 0
        if self.current_epoch == self.hparams.contrast_epoch:
            # init the parameter of key encoder
            for param_q, param_k in zip(self.basic_model.parameters(), self.key_encoder.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
            self.start_contrast = True

    def on_train_start(self) -> None:
        self.queue = nn.functional.normalize(self.queue, dim=1)
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--meta_lr_rate", default=0.01, type=float)
        parser.add_argument("--meta_weight_decay", default=0.001, type=float)
        parser.add_argument("--cls_emb_dim", default=128, type=int)
        parser.add_argument("--gw_hidden_dim", default=256, type=int)
        parser.add_argument("--is_deeper_weight", action='store_true')
        parser.add_argument("--gw_dropout", default=0.3, type=float)
        parser.add_argument("--gradient_steps", default=1, type=int,
                            help='Number of look-ahead gradient steps for meta-gradient (default: 1)')
        parser.add_argument("--is_kstep_meta", action='store_true', help='whether utilize Guoqing s method in aaai')
        parser.add_argument("--pre_train_basic_epochs", default=0, type=int)
        parser.add_argument("--pre_train_meta_epochs", default=0, type=int)
        parser.add_argument("--num_cola_score_histograms", default=100, type=int)
        parser.add_argument("--num_read_score_histograms", default=100, type=int)
        parser.add_argument("--histogram_embed_size", default=-1, type=int,
                            help="-1 means use the raw cola/read score as the feature for meta-weight model")
        parser.add_argument("--is_score_embed", action='store_true',
                            help='whether utilize the auxiliary score like readability/cola for the meta-weight model')
        parser.add_argument("--is_pre_meta_metric", action='store_true',
                            help='whether use the l1 distance for pre-training')
        parser.add_argument("--weight_output_dim", default=1, type=int,
                            help='if > 1 will apply the logits weight， '
                                 'this should match the number of class')
        parser.add_argument("--weight_scale", default=1.0, type=float,
                            help='')

        # contrastive learning related setting
        parser.add_argument("--is_sample_aligned", action="store_true")
        parser.add_argument("--is_weight_contrast", action="store_true")
        parser.add_argument("--is_supervised_contrast", action="store_true")
        parser.add_argument("--is_time_prior", action="store_true")
        parser.add_argument("--is_feature_normalized", action="store_true")
        parser.add_argument("--is_debug_label_flip", action="store_true")
        parser.add_argument("--num_neg", default=256, help='number of negative elements stored in memory bank')
        parser.add_argument("--momentum", default=0.99, help='momentum for key encoder update')
        parser.add_argument("--temperature", default=1.0)
        parser.add_argument("--base_temperature", default=1.0)
        parser.add_argument("--lambda_inner_contrast", default=1.0)
        parser.add_argument("--lambda_intra_contrast", default=0.0)
        parser.add_argument("--contrast_epoch", default=0)
        parser.add_argument("--contrast_patience", default=10)
        parser.add_argument("--ttu", default=50, help="life time for memory bank")
        parser.add_argument("--pos_ratio", default=1)
        parser.add_argument("--queue_exclusive", action="store_false")
        parser.add_argument("--is_add_noise", action="store_true")
        parser.add_argument("--is_scheduler_noise", action="store_true")
        parser.add_argument("--aug_dropout", default=0.2, type=float)
        parser.add_argument("--scheduler_threshold", default=1e-2, type=float)
        return parser
