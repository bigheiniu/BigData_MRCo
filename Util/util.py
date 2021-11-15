from sklearn.metrics import accuracy_score, f1_score, \
    recall_score, precision_score, confusion_matrix

import numpy as np
import json
import logging
from multiprocessing import Process
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import os
from glob import glob
import yaml
from Util.tflogs2pandas import compare_hyper

def load_json_file(file_name):
    data = []
    with open(file_name, 'r') as f1:
        for line in f1.readlines():
            data.append(json.loads(line))
    return data


def seperate_score(predict_Y, test_Y, average):
    f1 = f1_score(y_pred=predict_Y, y_true=test_Y,average=average)
    cf_m = confusion_matrix(y_pred=predict_Y, y_true=test_Y)
    precision = precision_score(y_true=test_Y, y_pred=predict_Y, average=average)
    recall = recall_score(y_true=test_Y, y_pred=predict_Y, average=average)
    return f1, precision, recall

def evaluation(logits, test_Y, metric):
    is_clf = type(test_Y[0]) == np.int64
    preds = np.argmax(logits, axis=1) if is_clf else np.squeeze(logits)
    result = metric.compute(predictions=preds, references=test_Y)
    if is_clf:
        predict_Y = np.argmax(logits, axis=1)
        acc = accuracy_score(y_pred=predict_Y, y_true=test_Y)
        try:
            # positive f1:
            average = "binary"
            pos_f1, pos_precision, pos_recall = seperate_score(predict_Y, test_Y, average)
            # negative f1:
            predict_Y = 1 - predict_Y
            test_Y = 1 - test_Y
            neg_f1, neg_precision, neg_recall = seperate_score(predict_Y, test_Y, average)
            result = {"acc": acc,
                    "positive_f1": pos_f1, "pos_recall": pos_recall, "pos_precision": pos_precision,
                    "neg_f1": neg_f1, "neg_recall": neg_recall, "neg_precision": neg_precision, **result}

        except:
            average = "micro"
            f1, precision, recall = seperate_score(predict_Y, test_Y, average)
            result = {"acc": acc, "f1": f1, "recall": recall, "precision": precision, **result}
    # placeholder for validation model selection
    else:
        # for sts-b task utilize pearson as the value for model selection
        result['acc'] = result['pearson']
    return result


def multiprocess_function(num_process, function_ref, args):
    jobs = []
    logging.info("Multiprocessing function %s started..." % function_ref.__name__)
    print("Multiprocessing function %s started..." % function_ref.__name__)

    for idx in range(num_process):
        process = Process(target=function_ref, args=(idx,) + args)
        process.daemon = True
        jobs.append(process)
        process.start()

    for i in range(num_process):
        jobs[i].join()

    logging.info("Multiprocessing function %s completed..." % function_ref.__name__)
    print("Multiprocessing function %s completed..." % function_ref.__name__)

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def rename_label(list_dict, task_name):
    return_list = []
    for i in list_dict:
        if "class" in i:
            i['label'] = i['class']
        if task_name == "cola" and "text" in i:
            i['sentence'] = i['text']

        return_list.append(i)
    return return_list
def transformer_load_dataset(path, task_name, aug_path=None, use_test=False, **kwargs):
    train_aug = rename_label(load_json_file(aug_path.format(task_name)), task_name)
    score_key = [i for i in train_aug[0].keys() if 'score' in i.lower() or 'readability' in i.lower()]
    read_score_interval = list(map(float, kwargs['readability'].split(",")))
    train_aug = [i for i in train_aug if all([read_score_interval[1] > i[key] > read_score_interval[0]
                                              for key in score_key])]
    train_raw = rename_label(load_json_file(path.format(task_name, "train")), task_name)
    train_all = pd.DataFrame(train_raw + train_aug)
    train_all = Dataset.from_pandas(train_all)
    val = load_dataset('glue', task_name)['validation']
    data_dict = {"train":train_all, 'validation': val}
    if use_test:
        test_data = pd.DataFrame(rename_label(load_json_file(path.format(task_name, "test")), task_name))
        test_data = Dataset.from_pandas(test_data)
        data_dict['test'] = test_data
    data = DatasetDict(data_dict)
    return data


def read_yaml(config_path):
    with open(config_path) as f1:
        docs = yaml.load_all(f1, Loader=yaml.FullLoader)
        doc = next(docs)
    return doc

def filter_already_finished(log_dir, search_args_list, is_increase_version_no=False, uninterested=('gpus_per_trial', 'version_no',
                                                                                                   'path')):
    # folder contains checkpoints folder is interrupted by the oom or other reasons
    already_yml_list = []
    if os.path.exists(log_dir) is False:
        return search_args_list
    for folder_path in os.listdir(log_dir):
        log_path = os.path.join(log_dir, folder_path)
        is_finished = len(glob(log_path + "/checkpoints")) == 0
        if is_finished:
            yml_path = glob(log_path+"/*.yaml")[0]
            already_yml_list.append(read_yaml(yml_path))
    increase_count = len(os.listdir(log_dir))
    keep_args = []
    for args in search_args_list:
        flag = True
        for index, i in enumerate(already_yml_list):
            if compare_hyper(dict1=args, dict2=i, hyper=uninterested):
                already_yml_list.pop(index)
                flag = False
                break
        if flag:
            if is_increase_version_no:
                args['version_no'] += increase_count
            keep_args.append(args)
    return keep_args












