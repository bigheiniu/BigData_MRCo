import torch
from torch.utils.data import dataset
from transformers import RobertaTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=4)


class OfflineAugmentDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, data_flag):
        self.hparams = hparams
        self.data_flag = data_flag
        self.is_roberta = getattr(self.hparams, "is_roberta", False)
        # train_clean_data, train_meta_data, train_aug_data
        self.two_sentence = False
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        if self.data_flag == "train":
            '''
            load training dataset
            '''
            # instance, label
            train_data = self.load_json_file(hparams.train_data_path)
            train_aug_data = self.load_json_file(hparams.train_aug_data_path)
            # manually filter the low quality augmented data
            # select interested augmentation method
            # ATTENTION: Pad the sample with the readability score and cola score
            if 'readability' not in train_aug_data[0].keys():
                read_score_label = [i for i in train_aug_data[0].keys() if "Score" in i]
                if len(read_score_label) == 0:
                    train_aug_data = [{**i, 'readability': 10, 'cola': 0.5} for i in train_aug_data]
                    train_data = [{**i, 'readability': 10, 'cola': 0.5} for i in train_data]
                elif len(read_score_label) == 1:
                    train_aug_data = [{**i, 'readability': i[read_score_label[0]], 'cola': 0.5} for i in train_aug_data]
                    train_data = [{**i, 'readability': 10, 'cola': 0.5} for i in train_data]
                else:
                    # TODO: this is placeholder to be compatiable with the model
                    train_aug_data = [{**i, 'readability': 10, 'readability1': i[read_score_label[0]],
                                       'readability2': i[read_score_label[1]], 'cola': 0.5} for i in train_aug_data]
                    train_data = [{**i, 'readability': 10, 'cola': 0.5, "encode":i['encode']} for i in train_data]
                    self.two_sentence = True

            if "label" in train_data[0].keys():
                train_aug_data = [{**i, "class": i['label']} for i in train_aug_data]
                train_data = [{**i, 'class': i['label']} for i in train_data]

                # stsb is not a binary task
            train_aug_data = self.filter_mechanism(train_aug_data, self.hparams.aug_method,
                                                   self.hparams.read_score_interval)

            train_data = pd.DataFrame(train_data)
            train_aug_data = pd.DataFrame(train_aug_data)
            # encode the text
            if self.two_sentence:
                def get_two_encode(data):
                    data['encode1'] = data['sentence1'].parallel_apply(
                            lambda x: tokenizer.encode(x + " </s></s> ", max_length=128, padding="do_not_pad",
                                                   truncation=True)[:-1])
                    data['encode2'] = data['sentence1'].parallel_apply(
                        lambda x: tokenizer.encode(x, max_length=128, padding="do_not_pad", truncation=True)[1:])
                    data['encode'] = data[['encode1', 'encode2']].agg(lambda x: x[0] + x[1], axis=1)
                    # there is a bug here cause mask_token_id
                    data['encode'] = data['encode'].apply(
                        lambda x: x[:256] + [tokenizer.pad_token_id] * max(256 - len(x), 0))
                    data = data.to_dict(orient="records")
                    return data
                train_data = get_two_encode(train_data)
                train_aug_data = get_two_encode(train_aug_data)

            else:
                train_data['encode'] = train_data['text'].parallel_apply(
                        lambda x: tokenizer.encode(x, max_length=128, padding="max_length", truncation=True))
                train_aug_data['encode'] = train_aug_data['text'].parallel_apply(
                    lambda x: tokenizer.encode(x, max_length=128, padding="max_length", truncation=True))
                train_data = train_data.to_dict(orient="records")
                train_aug_data = train_aug_data.to_dict(orient="records")

            assert set([len(i['encode']) for i in train_aug_data]).pop() == set([len(i['encode']) for i in train_data]).pop()
            if self.two_sentence and self.is_roberta:
                train_aug_data = [{**i, "encode": i['encode']} for i in train_aug_data]
                train_data = [{**i, "encode": i['encode']} for i in train_data]

            if len(train_aug_data) == 0:
                # hard filter cause no augmented data left.
                exit(-1)
            train_clean_data, train_meta_data, train_aug_data, train_clean_index = \
                self.get_train_samples(train_data,
                                       train_aug_data,
                                       hparams.train_count,
                                       hparams.clean_count)

            self.is_sample_aligned = getattr(hparams, "is_sample_aligned", False) or getattr(hparams,
                                                                                             "is_contrastive_learning",
                                                                                             False)
            if self.is_sample_aligned:
                # aligned the raw samples and augmented samples for contrastive learning
                # In the length_align only expand the meta-learning samples.
                train_clean_data, train_aug_data = self.sample_align(train_clean_data, train_aug_data)

            self.train_clean_data, self.train_meta_data, self.train_aug_data, self.train_clean_index = \
                self.length_align(train_clean_data, train_meta_data, train_aug_data, train_clean_index)

            if getattr(self.hparams, 'is_debug_label_flip', False):
                labels = np.array([i['class'] for i in self.train_aug_data], dtype=np.long)
                flipped_label_list = self.flip_debug(labels)
                self.train_aug_data = [{**i, "class": flip_label} for i, flip_label in zip(train_aug_data,
                                                                                           flipped_label_list)]
            self.clean_data = self.train_clean_data
            # print("train", len(self.clean_data))

        # val_clean_data
        elif self.data_flag == 'val':
            self.clean_data = self.load_json_file(hparams.val_data_path)
            if "label" in self.clean_data[0].keys():
                self.clean_data = [{**i, 'class': i['label']} for i in self.clean_data]
            if self.two_sentence and self.is_roberta:
                self.clean_data = [{**i, "encode": i['encode']} for i in self.clean_data]

            clean_data = pd.DataFrame(self.clean_data)
            if self.two_sentence is False:
            # if "text" in clean_data.columns:
                clean_data['encode'] = clean_data['text'].parallel_apply(
                    lambda x: tokenizer.encode(x, max_length=128, padding="max_length", truncation=True))
            else:
                clean_data['encode1'] = clean_data['sentence1'].parallel_apply(
                    lambda x: tokenizer.encode(x + " </s></s> ", max_length=128, padding="do_not_pad",
                                               truncation=True)[:-1])
                clean_data['encode2'] = clean_data['sentence1'].parallel_apply(
                    lambda x: tokenizer.encode(x, max_length=128, padding="do_not_pad", truncation=True)[1:])
                clean_data['encode'] = clean_data[['encode1', 'encode2']].agg(lambda x: x[0] + x[1], axis=1)
                # there is a bug here cause mask_token_id
                clean_data['encode'] = clean_data['encode'].apply(
                    lambda x: x[:256] + [tokenizer.mask_token_id] * max(256 - len(x), 0))
            # print('val', len(self.clean_data))
            self.clean_data = clean_data.to_dict(orient="records")

        else:
            self.clean_data = self.load_json_file(hparams.test_data_path)
            if "label" in self.clean_data[0].keys():
                self.clean_data = [{**i, 'class': i['label']} for i in self.clean_data]
            if self.two_sentence and self.is_roberta:
                self.clean_data = [{**i, "encode": i['encode']} for i in self.clean_data]
            clean_data = pd.DataFrame(self.clean_data)
            if "text" in clean_data.columns:
                clean_data['encode'] = clean_data['text'].parallel_apply(
                    lambda x: tokenizer.encode(x, max_length=128, padding="max_length", truncation=True))
            elif "sentence1" in clean_data.columns:
                clean_data['encode1'] = clean_data['sentence1'].parallel_apply(
                    lambda x: tokenizer.encode(x + " </s></s> ", max_length=128, padding="do_not_pad",
                                               truncation=True)[:-1])
                clean_data['encode2'] = clean_data['sentence1'].parallel_apply(
                    lambda x: tokenizer.encode(x, max_length=128, padding="do_not_pad", truncation=True)[1:])
                clean_data['encode'] = clean_data[['encode1', 'encode2']].agg(lambda x: x[0] + x[1], axis=1)
                # there is a bug here cause mask_token_id
                clean_data['encode'] = clean_data['encode'].apply(
                    lambda x: x[:256] + [tokenizer.mask_token_id] * max(256 - len(x), 0))
            # print('val', len(self.clean_data))
            self.clean_data = clean_data.to_dict(orient="records")
            # print('test', len(self.clean_data))


    def like_glue(self, encode):
        seperate_index = encode.index(0, 1, len(encode))
        encode = encode[:seperate_index] + [2, 2] + encode[seperate_index+1:]
        return encode

    def load_json_file(self, file_name):
        data = []
        with open(file_name, 'r') as f1:
            for line in f1.readlines():
                data.append(json.loads(line))
        return data

    def length_align(self, train_clean_data, train_meta_data, train_aug_data, train_clean_index=None):
        # TODO: different data loader?
        def length_align_helper(elements, required_length):
            if len(elements) > required_length:
                elements = elements[:required_length]
            else:
                elements = elements * int(required_length / len(elements))
                elements = elements + elements[:required_length - len(elements)]
            return elements
        if getattr(self.hparams, "no_aug_data", False):
            max_len = len(train_clean_data)
        elif getattr(self.hparams, "meta_only", False):
            max_len = len(train_meta_data)
        else:
            max_len = max(len(train_clean_data), len(train_meta_data), len(train_aug_data))
        train_clean_data = length_align_helper(train_clean_data, max_len)
        train_meta_data = length_align_helper(train_meta_data, max_len)
        train_aug_data = length_align_helper(train_aug_data, max_len)
        return train_clean_data, train_meta_data, train_aug_data, train_clean_index

    def sample_align(self, train_clean_data, train_aug_data):
        train_clean_df = pd.DataFrame(train_clean_data)
        train_aug_df = pd.DataFrame(train_aug_data)
        aligned_samples = pd.merge(train_clean_df, train_aug_df, how='inner', left_on='index', right_on='origin',
                                   suffixes=("_raw", '_aug'))
        clean_columns = [i for i in aligned_samples.columns if "_raw" in i]
        aug_columns = [i for i in aligned_samples.columns if "_aug" in i]
        train_clean_df = aligned_samples[clean_columns]
        train_aug_df = aligned_samples[aug_columns]
        train_clean_data = train_clean_df.rename(columns=lambda x: x.replace("_raw", "")).to_dict(orient="records")
        train_aug_data = train_aug_df.rename(columns=lambda x: x.replace("_aug", "")).to_dict(orient="records")
        return train_clean_data, train_aug_data

    def get_train_samples(self, train_data, train_aug_data, train_count, clean_count):
        # access limited number of training dataset
        # ATTENTION: the meta-train dataset has no overlap with the aug training dataset
        if train_count != -1:
            train_data, _ = train_test_split(train_data, stratify=[i['class'] for i in train_data],
                                             train_size=train_count)
        if clean_count != -1:
            train_clean_data, train_meta_data = \
                train_test_split(train_data, stratify=[i['class'] for i in train_data],
                                 train_size=clean_count)
        else:
            # baseline method does not need meta split
            train_clean_data = train_data
            train_meta_data = train_data
        train_clean_index = set([i['index'] for i in train_clean_data])
        # ATTENTION: this will enlarge the size of the training dataset,
        # pls check the length alignment function for details.
        train_aug_data = [i for i in train_aug_data if i['origin'] in train_clean_index]

        return train_clean_data, train_meta_data, train_aug_data, train_clean_index

    def filter_mechanism(self, train_aug_data, aug_method, read_score_interval):
        # TODO: filter low quality samples
        if aug_method != "all":
            aug_method = set(aug_method.split(","))
            train_aug_data = [i for i in train_aug_data if i['aug_method'] in aug_method]
        read_score_interval = read_score_interval.split("+")
        for i in range(len(read_score_interval)):
            try:
                read_score_interval[i] = float(read_score_interval[i])
            except:
                read_score_interval[i] = float(read_score_interval[i][1:])

        if self.two_sentence is False:
            train_aug_data = [i for i in train_aug_data
                              if (float(read_score_interval[0]) < i['readability'] < float(read_score_interval[1]))]
        else:
            train_aug_data = [i for i in train_aug_data
                              if (float(read_score_interval[0]) < i['readability1'] < float(read_score_interval[1])) and
                              (float(read_score_interval[0]) < i['readability2'] < float(read_score_interval[1]))
                              ]

        return train_aug_data

    def flip_debug(self, y_aug):
        # to see whether the meta_weight function can identify the flip label(noise samples)
        flip_pro = getattr(self.hparams, "debug_flip_pro", 0.4)
        flip_indicator = np.random.binomial(1, flip_pro, len(y_aug))
        flipped_label = np.where(flip_indicator == 1, 1 - y_aug, y_aug)
        return flipped_label

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, item):
        if self.data_flag == 'train':
            return torch.tensor(self.train_clean_data[item]['encode']), torch.tensor(
                self.train_clean_data[item]['class']), \
                   torch.tensor(self.train_clean_data[item]['readability']), torch.tensor(
                self.train_clean_data[item]['cola']), \
                   torch.tensor(self.train_meta_data[item]['encode']), torch.tensor(
                self.train_meta_data[item]['class']), \
                   torch.tensor(self.train_aug_data[item]['encode']), torch.tensor(self.train_aug_data[item]['class']), \
                   torch.tensor(self.train_aug_data[item]['readability']), torch.tensor(
                self.train_aug_data[item]['cola'])


        else:
            return torch.tensor(self.clean_data[item]['encode']), torch.tensor(self.clean_data[item]['class'])


def expand_list(elements, max_lenth):
    elements = elements * int(max_lenth / len(elements))
    elements = elements + elements[:max_lenth - len(elements)]
    return  elements

def collate_fn(batch):
    clean_encode, clean_label, aug_data, aug_label = zip(*batch)
    max_aug_samples = max([len(i) for i in aug_data])
    # pad the augmented samples
    aug_data_pad = [expand_list(i, max_aug_samples) for i in aug_data]
    aug_label_pad = [expand_list(i, max_aug_samples) for i in aug_label]
    return torch.tensor(clean_encode), torch.tensor(clean_label), torch.tensor(aug_data_pad), torch.tensor(aug_label_pad)
