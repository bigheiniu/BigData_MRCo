import pytorch_lightning as pl
import shutil
from pytorch_lightning.loggers import TensorBoardLogger
from Trainer import BaselineTrainer, AlignedMetaContrastTrainer
from argparse import ArgumentParser, Namespace
from Util import multiprocess_function, chunkify
from Util.read_configuration import read_yaml
from itertools import product
import logging
import traceback
import time
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from Util.util import filter_already_finished
import os


def nn_clf_method(idx, hparams_chunk):
    print("Process %d started"%idx)
    hparams_list = hparams_chunk[idx]
    for hparams in hparams_list:
        # try:
        pl.seed_everything(hparams.random_seed)
        model, auto = activate_model_init(hparams)
        logger = TensorBoardLogger('glue_tb_logs', name=hparams.log_name, version=hparams.version_no)
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            save_top_k=1,
            mode='max')
        if hparams.is_fp16:
            trainer = pl.Trainer(
                max_epochs=hparams.epochs,
                gpus=hparams.gpus,
                logger=logger,
                callbacks=[checkpoint_callback],
                gradient_clip_val=getattr(hparams, "gradient_clip_val", 0),
                precision=16,
                amp_level="O1",
                # terminate_on_nan=True
            )
        else:
            trainer = pl.Trainer(
                max_epochs=hparams.epochs,
                gpus=hparams.gpus,
                logger=logger,
                callbacks=[checkpoint_callback],
                gradient_clip_val=getattr(hparams, "gradient_clip_val", 0),
                # terminate_on_nan=True
            )
        trainer.fit(model)
        # utilize the last model
        trainer.test(model)
        trainer.test()
        shutil.rmtree(logger.log_dir + "/checkpoints")
        print("Success finish {} at Process {}".format(hparams.version_no, idx))
    print("Process %d stopped" % idx)

def activate_model_init(hparams):
    if hparams.model_name == 'baseline':
        model = BaselineTrainer(hparams)
        auto = True
    else:
        model = AlignedMetaContrastTrainer(hparams, is_half=getattr(hparams, "is_fp16", False))
        auto = False
    return model, auto

num_labels_dict = {
    "sst2": 2,
    "stsb": 1,
    'cola': 2,
    'rte': 2,
    'qnli':2,
    'mrpc':2,
    'wnli':2,
    'mnli':3,
    'qqp':2,
}
def hyper_monitor(hparams):
    tune_config, _, general_config = read_yaml(hparams.model_config_path)
    tune_config = tune_config.items()
    train_name = hparams.special_tag + general_config['model_name'] + "_" + hparams.task_name
    # set up the data path
    if "mnli" in hparams.task_name:
        general_config['train_data_path'] = general_config[
                                                'base_path'] + "/" + "mnli" + "/" + "train_data.json"
        general_config['train_aug_data_path'] = general_config[
                                                    'base_path'] + "/" + "mnli" + "/" + "all_aug_data.json"
        if "mis" in hparams.task_name:
            general_config['val_data_path'] = general_config['base_path'] + "/" + "mnli" + "/" + "dev_mismatched_data.json"
        else:
            general_config['val_data_path'] = general_config[
                                                  'base_path'] + "/" + "mnli" + "/" + "dev_matched_data.json"

        general_config['class_num'] = num_labels_dict["mnli"]
        hparams.task_name = "mnli"
    else:
        general_config['train_data_path'] = general_config['base_path'] + "/" + hparams.task_name + "/" + "train_data.json"
        general_config['val_data_path'] = general_config['base_path'] + "/" + hparams.task_name + "/" + "dev_data.json"
        general_config['train_aug_data_path'] = general_config['base_path'] + "/" + hparams.task_name + "/" + "all_aug_data.json"

        general_config['class_num'] = num_labels_dict[hparams.task_name]

    if hparams.task_name != 'sst2':
        general_config['test_data_path'] = general_config['val_data_path']
    else:
        general_config['test_data_path'] = general_config[
                                               'base_path'] + "/" + hparams.task_name + "/" + "test_data.json"

    setattr(hparams, "log_name", train_name)
    parameter_name = [i[0] for i in tune_config]
    parameter_value = [i[1]['grid_search'] for i in tune_config]
    #print(type(parameter_value), len(parameter_value), parameter_value)
    parameter_search = list(product(*parameter_value))
    parameter_search = [{**{name: value for name, value in zip(parameter_name, i)}, **general_config, **vars(hparams),
                         "version_no": index} for index, i
                        in enumerate(parameter_search)]
    parameter_search = filter_already_finished(log_dir=os.path.join("glue_tb_logs/", hparams.log_name),
                                               search_args_list=parameter_search, is_increase_version_no=True)
    parameter_search = [Namespace(**i) for i in parameter_search]
    print("There are %d trials" % len(parameter_search))
    num_process = int(1 / hparams.gpus_per_trial)
    parameter_search_chunk = chunkify(parameter_search, num_process)
    try:
        multiprocess_function(num_process, function_ref=nn_clf_method, args=(parameter_search_chunk, ))
    except Exception as e:
        logging.error(e)
        print("***ERROR***")
        print(traceback.format_exc())
        print(str(e))


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("--random_seed", default=123, type=int)
    args.add_argument("--model_config_path", default="./config/contrastive_roberta_mnli.yml")

    args.add_argument("--special_tag", default="")
    args.add_argument("--gpus", default=1, type=int)
    #args.add_argument("--epochs", default=30, type=int)
    args.add_argument("--gpus_per_trial", default=0.25, type=float)
    args.add_argument("--task_name", type=str, default="sst2")
    args.add_argument("--is_fp16", action='store_true')
    args = args.parse_args()
    hyper_monitor(args)

