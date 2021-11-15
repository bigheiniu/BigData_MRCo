import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, RobertaModel, RobertaConfig, RobertaForSequenceClassification
#
class DistilBERT_Text(nn.Module):
    def __init__(self, args):
        super(DistilBERT_Text, self).__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = nn.Linear(args.hidden_size, args.hidden_size)
        self.dropout = nn.Dropout(args.cnn_drop_out)
        C = args.class_num
        self.fc1 = nn.Linear(args.hidden_size, C)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None, **kwargs):
        distilbert_output = self.model(x)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        hidden = self.dropout(pooled_output)  # (bs, dim)
        logit = self.fc1(hidden)
        output = (hidden, logit,)
        if y is not None:
            loss = self.loss(logit, y)
            output += (loss,)
        return output

class RoBERTa_Text(nn.Module):
    def __init__(self, args):
        super(RoBERTa_Text, self).__init__()
        self.args = args
        config = RobertaConfig.from_pretrained(
            'roberta-base', num_labels=args.class_num, finetuning_task=args.task_name, output_hidden_states=True
        )

        if self.args.is_debug:
            config.num_hidden_layers = 2
            config.hidden_size = 36

        self.D = config.hidden_size
        C = args.class_num

        new_config = DummyConfig({'hidden_size':self.args.hidden_size, 'hidden_dropout_prob': config.hidden_dropout_prob, 'num_labels':config.num_labels})
        if self.args.is_debug:
            self.model = RobertaModel(config=config, add_pooling_layer=False)
        else:
            self.model = RobertaModel.from_pretrained('roberta-base', config=config, add_pooling_layer=False)
        # self.model = RobertaModel(config=config, add_pooling_layer=False)
        self.fc2 = nn.Linear(self.D, self.args.hidden_size)
        # self.fc1 = nn.Linear(self.args.hidden_size, C)
        self.fc1 = RobertaClassificationHead(new_config)

        if args.class_num == 1:
            self.loss_no_reduction = nn.MSELoss(reduction="none")
            self.loss_reduction = nn.MSELoss()
        else:
            self.loss_no_reduction = nn.CrossEntropyLoss(reduction="none")
            self.loss_reduction = nn.CrossEntropyLoss()

    def forward(self, x, y=None, is_reduction=True, **kwargs):
        device = x.device
        # lazy make the attention mask
        attention_mask = torch.where(x == 1, torch.zeros_like(x), torch.ones_like(x)).float().to(device)
        outputs = self.model(input_ids=x, attention_mask=attention_mask)
        # first token of the last layer
        hidden_states = outputs.hidden_states[-1][:, 0, :]
        hidden = self.fc2(hidden_states)
        logit = self.fc1(hidden)
        output = (hidden, logit,)
        if y is not None:
            if is_reduction:
                loss = self.loss_reduction(logit, y)
            else:
                loss = self.loss_no_reduction(logit, y)
            output += (loss,)
        return output

class DummyConfig(object):
    """docstring for DummyConfig"""
    def __init__(self, arg):
        super(DummyConfig, self).__init__()
        self.hidden_size = arg['hidden_size']
        self.hidden_dropout_prob = arg['hidden_dropout_prob']
        self.num_labels = arg['num_labels']
        

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
        # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CNN_Text(nn.Module):
    def __init__(self, args, use_roberta_wordembed=True):
        super(CNN_Text, self).__init__()
        self.args = args
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = list(map(int, args.kernel_sizes.split(",")))

        roberta_config = RobertaConfig.from_pretrained("roberta-base")
        V = roberta_config.vocab_size
        D = roberta_config.hidden_size
        if use_roberta_wordembed:
            roberta_model = RobertaModel.from_pretrained("roberta-base", config=roberta_config)
            embedding_weight = roberta_model.get_input_embeddings().weight
            # ATTENTION: the word embedding is not freezed
            is_freeze = getattr(args, "is_freeze", False)
            self.embed = nn.Embedding(V, D).from_pretrained(embedding_weight, freeze=is_freeze)
            del roberta_model
            del embedding_weight
        else:
            self.embed = nn.Embedding(V, D)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.cnn_drop_out)

        self.fc1 = nn.Linear(self.args.hidden_size, C)
        self.fc2 = nn.Linear(len(Ks) * Co, self.args.hidden_size)
        self.class_num = args.class_num
        if self.class_num == 1:
            self.loss_no_reduction = nn.MSELoss(reduction="none")
            self.loss_reduction = nn.MSELoss()
        else:
            self.loss_no_reduction = nn.CrossEntropyLoss(reduction="none")
            self.loss_reduction = nn.CrossEntropyLoss()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, y=None, is_reduction=True, **kwargs):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        hidden = self.fc2(x)
        logit = self.fc1(hidden)  # (N, C)
        output = (hidden, logit,)
        if y is not None:
            if is_reduction:
                loss = self.loss_reduction(logit, y)
            else:
                loss = self.loss_no_reduction(logit, y)
            output += (loss, )
        return output



