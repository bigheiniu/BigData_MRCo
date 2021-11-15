from transformers import RobertaTokenizer
import pandas as pd
import torch
import json
from glob import glob
import os
from tqdm import tqdm


def tokenize_length(data,max_length, oupput_file, is_multi=False):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    if is_multi is False:
        data['encode'] = data['text'].apply(lambda x:
                                           tokenizer.encode(x,
                                                            max_length=max_length,
                                                            truncation=True,
                                                            padding='max_length'
                                                            )
                                           )
    else:
        data['encode1'] = data['text1'].apply(lambda x:
                                            tokenizer.encode(x,
                                                             max_length=max_length,
                                                             truncation=True,
                                                             padding='max_length'
                                                             )
                                            )
        data['encode2'] = data['text2'].apply(lambda x:
                                              tokenizer.encode(x,
                                                               max_length=max_length,
                                                               truncation=True,
                                                               padding='max_length'
                                                               )
                                              )
        data['encode'] = data[['encode1', "encode2"]].agg(lambda x: x[0]+x[1], axis=1)


    data_json = data.to_dict(orient='records')
    with open(oupput_file, 'w') as f1:
        for i in data_json:
            f1.write(json.dumps(i)+"\n")
            f1.flush()


def read_json_file(file):
    data = json.load(open(file, 'r'))
    data.pop("idx", None)
    label = data['label']
    data_sample = []
    for key in label:
        th = [int(key)]
        for _, ele in data.items():
            th.append(ele[key])
        data_sample.append(th)

    return data_sample


def read_mnli_task(json_file, task_name, train_type):
    data = read_json_file(json_file)
    output_file = './{}/{}_mismatched_data.json'.format(task_name, train_type)
    # output_file = "/".join(json_file.split("/")[:-1] + ["encode" + json_file.split("/")[-1]])
    if len(data[0]) == 4 and train_type == "train":
        data = pd.DataFrame(data, columns=['index','label','text','readability'])
        tokenize_length(data, 128, output_file)
    elif len(data[0]) == 3:
        data = pd.DataFrame(data, columns=['index', 'label', 'text'])
        tokenize_length(data, 128, output_file)
    elif len(data[0]) == 4 and train_type != "train":
        # data = pd.DataFrame(data, columns=['index','label','sentence1', 'sentence2'])
        data = pd.DataFrame(data, columns=['index', 'sentence1', 'label', 'sentence2'])
        if "sentence1" in data.columns:
            data['text1'] = data['sentence1']
            data['text2'] = data['sentence2'].apply(lambda x: " </s> "+ x)
            # data['text'] = data.apply(lambda x: x['sentence1'] + " </s> " + x['sentence2'], axis=1)
        else:
            data['text1'] = data['question1']
            data['text2'] = data['question2'].apply(lambda x: " </s> "+ x)
            # data['text'] = data.apply(lambda x: x['question1'] + " </s> " + x['question2'], axis=1)
        tokenize_length(data, 128, output_file, is_multi=True)
    elif len(data[0]) == 6:
        data = pd.DataFrame(data, columns=['index', 'sentence1','label', 'sentence2', 'score1', 'score2'])
        if "sentence1" in data.columns:
            # data['text'] = data.apply(lambda x: x['sentence1'] + " </s> " + x['sentence2'], axis=1)
            data['text1'] = data['sentence1']
            data['text2'] = data['sentence2'].apply(lambda x: " </s> "+ x)
        else:
            data['text1'] = data['question1']
            data['text2'] = data['question2'].apply(lambda x: " </s> " + x)
            # data['text'] = data.apply(lambda x: x['question1'] + " </s> " + x['question2'], axis=1)
            # data['text'] = data.apply(lambda x: x['question1'] + " </s> " + x['question2'], axis=1)
        tokenize_length(data, 128 * 2, output_file, is_multi=True)

def read_task_single(task_name, base_dir):
    origin_train = None
    aug_data_all = pd.DataFrame()
    for file in glob(base_dir+"/_{}*".format(task_name)):
        if "train" in file:
            origin_train = read_json_file(file)
            if len(origin_train[0]) > 4:
                return 1
        elif "Aug" in file:
            aug_method = file.split("_")[-2]
            aug_data = pd.read_csv(file)
            aug_data['aug_method'] = aug_method
            aug_data_all = aug_data_all.append(aug_data, ignore_index=True)
    try:
        print("There are {} Aug Method for {}\n They are {}".format(len(aug_data_all['aug_method'].unique()), task_name,
                                                                    aug_data_all['aug_method'].unique()))
    except:
        print("ERROR AT {}".format(task_name))
    # merge)
    origin_train_join = pd.DataFrame([(i[0], i[1]) for i in origin_train], columns=['index', 'label'])
    aug_data = pd.merge(aug_data_all, origin_train_join, how='left', left_on='origin', right_on='index')
    save_dir = "./{}".format(task_name)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    aug_data = aug_data.rename(columns={"sentence": "text"})
    tokenize_length(aug_data, max_length=128, oupput_file="./{}/all_aug_data.json".format(task_name))
    origin_train_join.to_csv("./{}/train.csv".format(task_name), index=False)
    return None

def read_task_multiple(task_name, base_dir):
    origin_train = None
    aug_data_all_sent1 = pd.DataFrame()
    aug_data_all_sent2 = pd.DataFrame()
    for file in glob(base_dir+"/_{}*".format(task_name)):
        if "train" in file:
            origin_train = read_json_file(file)
            if len(origin_train[0]) < 5:
                return
        elif "Aug" in file:
            aug_method = file.split("_")[-2]
            aug_data = pd.read_csv(file)
            aug_data['aug_method'] = aug_method
            if "sentence1" in file or "question1" in file or "_question_" in file or "hypothesis" in file:
                aug_data_all_sent1 = aug_data_all_sent1.append(aug_data, ignore_index=True)
            elif "sentence2" in file or "question2" in file or "_sentence_" in file or "premise" in file:
                aug_data_all_sent2 = aug_data_all_sent2.append(aug_data, ignore_index=True)

    if len(aug_data_all_sent2) == 0:
        return
    # subsample each group
    aug_data_all = pd.merge(aug_data_all_sent1, aug_data_all_sent2, how='inner', on=['aug_method', 'origin'])
    aug_data_all.dropna(inplace=True, how='any')
    aug_data_all_group = aug_data_all.groupby(['aug_method', 'origin'])
    aug_data_all = sub_sample_group(aug_data_all_group, n=2)

    print("There are {} Aug Method for {}\n They are {}".format(len(aug_data_all['aug_method'].unique()), task_name,
                                                                aug_data_all['aug_method'].unique()))
    print("There are {} augmented samples, compared with raw {}".format(len(aug_data_all), len(origin_train)))
    origin_train_join = pd.DataFrame([(i[0], i[1]) for i in origin_train], columns=['index', 'label'])
    aug_data = pd.merge(aug_data_all, origin_train_join, how='left', left_on='origin', right_on='index')
    save_dir = "./{}".format(task_name)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    try:
        aug_data['text'] = aug_data.apply(lambda x: x['sentence1'] + "</s></s>" + x['sentence2'], axis=1)
    except:
        try:
            aug_data['text'] = aug_data.apply(lambda x: x['question1'] + "</s></s>" + x['question2'], axis=1)
        except:
            try:
                aug_data['text'] = aug_data.apply(lambda x: x['question'] + "</s></s>" + x['sentence'], axis=1)
            except:
                aug_data['text'] = aug_data.apply(lambda x: x['hypothesis'] + "</s></s>" + x['premise'], axis=1)
    tokenize_length(aug_data, max_length=128, oupput_file="./{}/all_aug_data.json".format(task_name), is_multi=True)
    # origin_train_join.to_csv("{}/train.csv".format(task_name), index=False)

def sub_sample_group(data_group, n=2):
    new_list = []
    for i in data_group:
        # since we keep the data, so here I did not set the seed
        sub_sample_group = i[1].sample(n=min(n, len(i[1])))
        new_list.append(sub_sample_group)
    return pd.concat(new_list, axis=0)

def read_mnli(file_list, output_file):
    data = pd.DataFrame()
    for file in file_list:
        h = read_json_file(file)
        h = pd.DataFrame(h, columns=["index", 'sentence1', 'label', 'sentence2'])
        data = data.append(h, ignore_index=True)
    data['text'] = data.apply(lambda x: x['sentence1'] + "</s></s>" + x['sentence2'], axis=1)
    tokenize_length(data, 50 * 2, output_file)


if __name__ == '__main__':
    base_dir = "./glue_dataset"

    for task in ['qqp', 'qqp', 'rte', 'mrpc', 'qnli']:
        read_task_single(task_name=task, base_dir=base_dir)

    for task in ['cola']:
        read_task_single(task, base_dir)

    for task in ['mnli']:
        for train_type in ['validation','test']:
            file = "./{}/_{}_{}_mismatched_.json".format(base_dir, task, train_type)
            train_type = train_type if train_type != "validation" else "dev"
            read_mnli_task(file, task, train_type)







