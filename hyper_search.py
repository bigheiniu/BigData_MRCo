import pytorch_lightning as pl
import shutil
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from HyperTune import hyper_parameter_selection
from glob import glob
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from argparse import Namespace
import torch
import os
os.environ["SLURM_JOB_NAME"] = "bash"
import ray
from Util import read_yaml
from Trainer import BaselineTrainer, MetaTrainer, AlignedMetaContrastTrainer
# utilize tune for hyper-parameter selection
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def activate_model_init(hparams):
    if hparams.model_name == 'baseline':
        model = BaselineTrainer(hparams)
    else:
        model = AlignedMetaContrastTrainer(hparams, is_half=hparams.is_fp16)
    return model


def train_model(config, method_name, haprams, num_gpus):
    tag = ""
    if type(haprams) is dict:
        haprams = Namespace(**{**haprams, **config})

    pl.seed_everything(haprams.random_seed)

    logger = TensorBoardLogger('tb_logs',
                               name=method_name + tag)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        save_top_k=1,
        mode='max')

    model = activate_model_init(haprams)
    automatic_optimization = True if haprams.model_name == 'baseline' else False
    # ATTENTION: automatic_optimization should set to True for Meta-Learning Model
    if haprams.is_fp16:
        trainer = pl.Trainer(
            max_epochs=haprams.epochs,
            callbacks=[checkpoint_callback],
            gpus=num_gpus,
            logger=logger, automatic_optimization=automatic_optimization,
            progress_bar_refresh_rate=0,
            precision=16,
            amp_level="O1"
        )
    else:
        trainer = pl.Trainer(
            max_epochs=haprams.epochs,
            callbacks=[checkpoint_callback],
            gpus=num_gpus,
            logger=logger, automatic_optimization=automatic_optimization,
            progress_bar_refresh_rate=0
        )

    trainer.fit(model)
    # trainer.test(model)
    # trainer.test()
    print("Log dir is "+logger.log_dir)
    shutil.rmtree(logger.log_dir+"/checkpoints")

def main(hparams):
    tune_config, search_hyper_parameters, general_config = read_yaml(hparams.model_config_path)
    general_config = {**vars(hparams), **general_config}
    train_name = general_config['special_tag'] + general_config['model_name']
    hyper_parameter_selection(tune_config, train_fn=train_model, search_hyper_parameters=search_hyper_parameters,
                              train_name=train_name, num_epochs=hparams.epochs, **general_config)

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--random_seed", default=123, type=int)
    args.add_argument("--model_config_path", default="./config/contrastive_roberta_mnli.yml")
    args.add_argument("--special_tag", default="")
    args.add_argument("--is_fp16", action="store_true")
    args.add_argument("--epochs", default=30, type=int)

    args = args.parse_args()

    ray.init(
        num_gpus=torch.cuda.device_count(),
        num_cpus=15,
    )

    for method_name in ['cnn']:
        main(args)
