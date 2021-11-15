import pytorch_lightning as pl
import shutil
from pytorch_lightning.loggers import TensorBoardLogger
from Trainer import MetaTrainer, MetaTrainerLast, AlignedMetaContrastTrainer
from argparse import ArgumentParser
from glob import glob
from Util import read_yaml

def nn_clf_method(method_name, hparams):
    # set the random seed for each experiment
    # hparams = BaselineTrainer.add_model_specific_args(hparams)
    hparams = AlignedMetaContrastTrainer.add_model_specific_args(hparams)
    hparams = hparams.parse_args()
    pl.seed_everything(hparams.random_seed)
    # model = MetaTrainer(hparams)
    is_half = getattr(hparams, "is_fp16", False)
    model = AlignedMetaContrastTrainer(hparams, is_half=is_half)
    logger = TensorBoardLogger('tb_logs', name=method_name + "_" + hparams.special_log_tag)
    trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=hparams.gpus,
            logger=logger,
            precision=16,
            amp_level="O1"
        )
    trainer.fit(model)
    # utilize the last model
    trainer.test(model)
    trainer.test()
    shutil.rmtree(logger.log_dir + "/checkpoints")

def evaluation(hparams, checkpoint_path):
    hparams = MetaTrainer.add_model_specific_args(hparams)
    hparams = hparams.parse_args()
    pl.seed_everything(hparams.random_seed)
    ckpt = glob("{}/*ckpt".format(checkpoint_path))[0]
    model = MetaTrainer.load_from_checkpoint(ckpt,  hparams=hparams)
    logger = TensorBoardLogger('tb_logs', name="cnn_eval" + "_" + hparams.special_log_tag)

    trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=hparams.gpus,
            logger=logger,
            automatic_optimization=False
        )

    trainer.test(model)



if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("--train_data_path", type=str, default="./Dataset/sst2/train_data.json")
    args.add_argument("--train_aug_data_path", type=str, default="./Dataset/sst2/all_aug_data.json")
    args.add_argument("--val_data_path", type=str, default="./Dataset/sst2/dev_data.json")
    args.add_argument("--test_data_path", type=str, default="./Dataset/sst2/test_data.json")
    args.add_argument("--clean_count", type=float, default=-1)
    args.add_argument("--train_count", type=float, default=-1)
    args.add_argument("--random_seed", default=123, type=int)
    args.add_argument("--basic_lr_rate", default=0.001, type=float)
    args.add_argument("--basic_weight_decay", default=0, type=float)
    # data augmentation
    args.add_argument("--no_aug_data", action="store_true")
    args.add_argument("--aug_method", default='all', type=str, choices=['all','CharswapAug','ChecklistAug','EasydataAug',
                                                                        'EmbeddingAug', 'WordnetAug'
                                                                        ])
    # 30-121.22 for the readability is preferred
    args.add_argument("--read_score_interval", default='0,200', type=str)
    args.add_argument("--cola_score_interval", default='-1,1.5', type=str)
    args.add_argument("--special_log_tag", default="", type=str)

    args.add_argument("--is_roberta", action="store_true")
    args.add_argument("--is_freeze", action="store_true")
    args.add_argument("--task_name", default="sst2", type=str)



    nn_model_args = args.add_argument_group("neural network ckpt parameters")
    nn_model_args.add_argument("--class_num", default=2, type=int)
    nn_model_args.add_argument("--kernel_num", default=100, type=int)
    nn_model_args.add_argument("--kernel_sizes", default="3,4,5", type=str)
    nn_model_args.add_argument("--cnn_drop_out", default=0.3, type=float)
    nn_model_args.add_argument("--hidden_size", default=64, type=int)
    nn_model_args.add_argument("--train_batch_size", default=32, type=int)
    nn_model_args.add_argument("--eval_batch_size", default=32, type=int)
    nn_model_args.add_argument("--epochs", default=30, type=int)
    nn_model_args.add_argument("--final_epochs", default=2, type=int)
    nn_model_args.add_argument("--gpus", default=1, type=int)

    nn_clf_method('cnn', args)

