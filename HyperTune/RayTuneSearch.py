import shutil
from functools import partial
from tempfile import mkdtemp
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

from ray.tune.integration.pytorch_lightning import TuneReportCallback
import torch
import os

def hyper_paramter_kits(num_epochs, search_hyper_parameters):
    callback = TuneReportCallback({
        "loss": "val_loss",
        "accuracy": "val_acc"
    }, on="validation_end")

    scheduler = ASHAScheduler(
        metric="val_acc",
        mode="max",
        max_t=100,
        grace_period=5,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=search_hyper_parameters,
        metric_columns=["loss", "accuracy", "training_iteration"])

    return callback, scheduler, reporter


def trial_dirname_creator(trail):
    return str(trail.trainable_name) + "_" + str(trail.trial_id)

def hyper_parameter_selection(config, train_fn, num_epochs, train_name, search_hyper_parameters,
                              **train_kwargs):
    gpus_per_trial = train_kwargs.get("gpus_per_trial", 0)
    cpus_per_trial = train_kwargs.get("cpus_per_trial", 1)
    callback, scheduler, reporter = hyper_paramter_kits(num_epochs, search_hyper_parameters)
    train_kwargs['gpus_per_trial'] = gpus_per_trial
    train_kwargs['cpus_per_trial'] = cpus_per_trial
    result = tune.run(
        partial(
            # method_name, haprams, config, num_gpus
            train_fn,
            num_gpus=1,
            method_name=train_kwargs['model_name'],
            haprams=train_kwargs,
            ),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        config=config,
        # default set for one turn
        scheduler=scheduler,
        progress_reporter=reporter,
        name=train_name,
        trial_dirname_creator=trial_dirname_creator,
        keep_checkpoints_num=1,
        checkpoint_score_attr="accuracy",
        # fail_fast=True,
    )
    # define anything you want
    best_trial = result.get_best_trial("accuracy", "max", "all")
    return best_trial

def evaluate(best_trial, trainer):
    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    trainer = trainer.load_from_checkpoint(checkpoint_path)
    # take the best checkpoint
    trainer.test()







