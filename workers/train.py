import os
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

from datasets import load_dataset
from models import load_model


# Train worker: worker for training segmentation task
def train_worker(args):
    kfold = 5 if args.kfold else 1

    for fold in range(kfold):
        if args.resume != -1:
            args.fold = args.resume
        else:
            args.fold = fold

        result = load_dataset(args, train=True)
        if len(result) == 3:
            train_dataset, val_dataset, collate_fn = result
        else:
            train_dataset, val_dataset = result
            collate_fn = None

        print('=> Preparing model for training')
        model = load_model(args)
        print('=> Model load finished')
        epoch = args.epoch
        save_path = os.path.join(args.save_name, str(args.seed))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            persistent_workers=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            persistent_workers=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )

        # optionally resume from a checkpoint
        model_path = None
        if args.resume == fold:
            model_path = "{}{}/model_checkpoint_{}.ckpt".format(args.save_name, str(args.seed), str(fold))
            if os.path.exists(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
            else:
                print("=> no checkpoint found at '{}'".format(model_path))
                model_path = None

        monitor = "Val/Avg_Dsc"
        mode = "max"

        checkpoint_best = ModelCheckpoint(
            dirpath=save_path,
            monitor=monitor,
            mode=mode,
            filename="model_best_{}".format(str(fold)),
        )

        # checkpoint_callback = ModelCheckpoint(
        #     dirpath=save_path,
        #     filename="model_checkpoint_{}".format(str(fold))
        # )

        checkpoint_last = ModelCheckpoint(
            dirpath=save_path,
            filename="model_last_{}".format(str(fold)),
            every_n_epochs=args.epoch,
        )

        find = True if args.model == 'medsam' else False
        trainer = Trainer(
            accelerator='gpu',
            devices=args.gpu,
            strategy=DDPStrategy(find_unused_parameters=find),
            logger=args.logger,
            callbacks=[checkpoint_best, checkpoint_last],
            max_epochs=epoch,
            log_every_n_steps=50
        )

        trainer.fit(model, train_loader, val_loader, ckpt_path=model_path)
