import torch
import torch.nn as nn
import numpy as np
class FullWeightModel(nn.Module):
    def __init__(self, hparams):
        super(FullWeightModel, self).__init__()
        self.hparams = hparams
        class_num = hparams.class_num
        hidden_size = hparams.hidden_size
        cls_emb_dim = hparams.cls_emb_dim
        gw_hidden_dim = hparams.gw_hidden_dim
        self.cls_emb = nn.Embedding(class_num, cls_emb_dim)
        hidden_size_input = hidden_size + cls_emb_dim

        # weight scaling
        self.weight_scale = self.hparams.weight_scale if hasattr(self.hparams, "weight_scale") else 1
        self.weight_output_dim = self.hparams.weight_output_dim if hasattr(self.hparams, "weight_output_dim") else 1
        # score embedding
        self.is_score_embed = self.hparams.is_score_embed if hasattr(self.hparams, "is_score_embed") else False
        if self.is_score_embed:
            self.readability_score_embed = AuxiliaryEmbedding(self.hparams.num_read_score_histograms,
                                                        self.hparams.histogram_embed_size,
                                                        self.hparams.read_score_interval)
            self.cola_score_embed = AuxiliaryEmbedding(self.hparams.num_cola_score_histograms,
                                                 self.hparams.histogram_embed_size,
                                                 self.hparams.cola_score_interval)

            hidden_size_input += 2 * self.hparams.histogram_embed_size

        if self.hparams.is_deeper_weight:
            self.ins_weight = nn.Sequential(
                nn.Linear(hidden_size_input, gw_hidden_dim),
                nn.Dropout(hparams.gw_dropout),
                nn.ReLU(),  # Tanh(),
                nn.Linear(gw_hidden_dim, gw_hidden_dim),
                nn.ReLU(),
                nn.Linear(gw_hidden_dim, self.weight_output_dim),
                nn.Sigmoid()
            )
        else:
            self.ins_weight = nn.Sequential(
                nn.Linear(hidden_size_input, gw_hidden_dim),
                nn.Dropout(hparams.gw_dropout),
                nn.ReLU(),  # Tanh(),
                nn.Linear(gw_hidden_dim, self.weight_output_dim),
                nn.Sigmoid()
            )

        self.pre_train_loss = nn.CrossEntropyLoss()


    def forward(self, x_feature, y_label, loss_s=None, cola_score=None, readability_score=None, y_logits=None, **kwargs):
        '''
        item_loss = 1 is just the placeholder
        '''
        x_feature = x_feature.detach()
        y_emb = self.cls_emb(y_label)
        hidden = torch.cat([y_emb, x_feature], dim=-1)
        weight = self.ins_weight(hidden) * self.weight_scale
        if y_logits is not None and self.weight_output_dim > 1:
            # make manipulation on the predicted logits
            y_logits = y_logits * weight
            loss_fn = nn.CrossEntropyLoss()
            loss_s = loss_fn(y_logits, y_label)
            return loss_s, weight
        elif loss_s is not None:
            log_loss_s = torch.mean(loss_s.detach())
            # if the weight are close to zero, loss_s will not represent any thing
            loss_s = torch.mean(weight * loss_s)
            return loss_s,  log_loss_s, weight
        return weight

    def pre_train(self, x_feature, y_label, cola_score=None, readability_score=None):
        # ATTENTION: The pre-train is for the instance level
        # binary classification
        x_feature = x_feature.detach()
        y_label_flip = 1 - y_label
        weight_positive = self.forward(x_feature, y_label, cola_score=cola_score, readability_score=readability_score)
        weight_negative = self.forward(x_feature, y_label_flip, cola_score=cola_score, readability_score=readability_score)

        if self.hparams.is_pre_meta_metric:
            # metric based pairwise learning to rank
            pre_train_loss = -torch.sum(weight_positive - weight_negative)
        else:
            # classification based pairwise learning to rank:
            prob = torch.cat([weight_negative, weight_positive], dim=1)
            expected_label = torch.ones_like(y_label)
            pre_train_loss = self.pre_train_loss(prob, expected_label)

        return pre_train_loss



class AuxiliaryEmbedding(nn.Module):
    def __init__(self, num_histogram, histogram_embed_size, decision_boundary):
        super(AuxiliaryEmbedding, self).__init__()
        # positional embedding of the histogram of the scores
        self.histogram_embed_size = histogram_embed_size
        if self.histogram_embed_size > 0:
            # use the raw score or the histogram embedding
            self.positional_embedding = nn.Embedding(num_histogram, histogram_embed_size)
            decision_boundary = [float(x) for x in decision_boundary.split(",")]
            self.histogram_step = (decision_boundary[1] - decision_boundary[0]) * 1. / num_histogram
            self.lower_boundary = decision_boundary[0]

    def forward(self, scores):
        if self.histogram_embed_size > 1:
            histogram_index = (scores - self.lower_boundary) / self.histogram_step
            histogram_index = histogram_index.long()
            score_embed = self.positional_embedding(histogram_index)
        else:
            # utilize the previous feature
            score_embed = scores
        return score_embed
