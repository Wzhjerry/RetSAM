""" Wrapper to train and test a medical image segmentation model. """
import argparse
import json
import os
from argparse import Namespace
import torch
import yaml
import ast
import wandb
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.fabric.utilities.seed import seed_everything
from workers.test import test_worker, combine_kfold
from workers.train import train_worker
from utils.train_utils import get_rank

torch.set_float32_matmul_precision("high")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    parser = argparse.ArgumentParser(description="2D Medical Image Segmentation")
    # multi-processing
    parser.add_argument(
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    # training configuration
    parser.add_argument(
        "-b",
        "--batch_size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size (default: 128), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--test_batch_size",
        default=12,
        type=int,
        metavar="N",
        help="inference mini-batch size (default: 1)",
    )
    parser.add_argument(
        "--epoch",
        default=100,
        type=int,
        metavar="N",
        help="training epoch (default: 100)",
    )
    parser.add_argument(
        "--resume",
        default=-1,
        type=int,
        metavar="N",
        help="resume from which fold (default: -1)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="ADAM beta")
    parser.add_argument(
        "--optimizer",
        default="ADAM",
        choices=("SGD", "ADAM", "RMSprop"),
        help="optimizer to use (SGD | ADAM | RMSprop)",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--pretrained", default="", help="pretrained model weights")
    parser.add_argument("--size", type=int, default=224, help="size of input image")
    parser.add_argument(
        "--dice_loss", action="store_true", help="using dice loss or not"
    )

    # model specifications
    parser.add_argument("--model", type=str, default="swin_multitask", help="model name (multitask only: swin_multitask or swin_multitask_final)")
    parser.add_argument("--out_channels", type=str, default="(2,3,2,4,6)", help="output channels")
    parser.add_argument("--patch_size", type=int, default=2, help="size of patch")
    parser.add_argument("--window_size", type=int, default=7, help="size of window")
    parser.add_argument("--depths", type=str, default="(2, 2, 2, 2)", help="depth of model")
    parser.add_argument("--num_heads", type=str, default="(3, 6, 12, 24)", help="Number of attention heads")
    parser.add_argument("--feature_size", type=int, default=24, help="size of feature embed dimension")
    
    # experiment configuration
    parser.add_argument("--task", type=str, default="vessel", help="task name")
    parser.add_argument("--save_results", action="store_true", help="save context results or not")
    parser.add_argument("--save_name", default="smoke", help="experiment name")
    parser.add_argument("--dataset", default="staining134", help="dataset name",)
    parser.add_argument("--dataset_name", default=None, help="explicit dataset folder name (if different from module name)")
    parser.add_argument("--train_csv_name", default="train.csv", help="CSV filename for training list")
    parser.add_argument("--kfold", action="store_true", help="5-fold cross-validation")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--description", default="", type=str, help="description of the experiment")
    parser.add_argument("--seed", type=int, default=436, help="global setting for random seed")
    parser.add_argument("--class_weights", type=str, default="(1, 1, 1.5, 0.8, 1.2)", help="class weights")

    # Wandb configuration
    parser.add_argument(
        "--log_name", default="", type=str, help="description of the wandb logger"
    )
    parser.add_argument("--tags", default=[], help="tags for wandb")
    parser.add_argument(
        "--id", type=str, default="", help="resume id for wandb"
    )
    parser.add_argument("--evaluate", action="store_true", help="evaluate only")

    args = parser.parse_args()

    args.out_channels = ast.literal_eval(args.out_channels)
    args.depths = ast.literal_eval(args.depths)
    args.num_heads = ast.literal_eval(args.num_heads)
    args.class_weights = ast.literal_eval(args.class_weights)
    for arg in vars(args):
        if vars(args)[arg] == "True":
            vars(args)[arg] = True
        elif vars(args)[arg] == "False":
            vars(args)[arg] = False

    dirname = "{}".format(args.save_name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    json.dump(args.__dict__, open(args.save_name + "/config.json", "w"), indent=2)

    if args.debug:
        print("=> Using local csv logger")
        args.logger = CSVLogger(
            save_dir=args.save_name,
            name=args.log_name,
            flush_logs_every_n_steps=50
        )
    else:
        if args.resume != -1:
            print("=> Resuming wandb logger")
            args.logger = WandbLogger(
                project='retsam',
                name=args.log_name,
                tags=args.tags,
                notes=args.description,
                id=args.id,
                resume=True,
            )
        else:
            print("=> Using wandb logger")
            args.logger = WandbLogger(
                project='retsam',
                name=args.log_name,
                tags=args.tags,
                notes=args.description,
            )
        if get_rank() == 0:
            args.logger.experiment.config.update(args, allow_val_change=True)
            wandb.run.log_code(".")

    seed_everything(args.seed, workers=True)
    if args.evaluate:
        print("=> Only evaluating")
        test_worker(args)
    else:
        print("=> Start segmentation training process")
        train_worker(args)
        print("=> Segmentation training process finished")

        print("=> Start testing segmentation model")
        test_worker(args)
        print("=> Segmentation model test finished")
    if args.kfold:
        print("=> Start combining kfold results")
        combine_kfold(args)
        print("=> Kfold results combined")


if __name__ == "__main__":
    main()
