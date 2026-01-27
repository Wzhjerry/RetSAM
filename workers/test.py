import os
import csv
import json
import numpy as np
import ast
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, top_k_accuracy_score, roc_auc_score
from datasets import load_dataset
from models import load_model


def test_worker(args):
    save_path = os.path.join(args.save_name, str(args.seed))

    folds = 5 if args.kfold else 1
    for fold in range(folds):
        args.fold = fold
        # args.dataset_folder = os.path.join(args.dataset_path, "images")
        result = load_dataset(args, train=False)
        model = load_model(args)
        
        # Check if dataset returns collate_fn
        if isinstance(result, tuple):
            test_dataset, collate_fn = result
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.test_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            test_dataset = result
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.test_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )

        # load corresponding model
        best_model_path = "{}/model_best_{}.ckpt".format(save_path, fold)
        # model.load_weights(best_model_path)
        model.init_weights(best_model_path)

        trainer = Trainer(accelerator="gpu", devices=1, num_nodes=1, logger=args.logger)
        trainer.test(model, dataloaders=test_loader)


def combine_kfold(args):
    save_path = args.save_name

    # Determine metrics based on the task
    metrics_keys = ["name", "dscs", "jacs"]

    metrics = {key: [] for key in metrics_keys}

    for fold in range(5):
        file_path = os.path.join(save_path, f"count_results_{fold}_{args.seed}.csv")
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in metrics_keys:
                    metrics[key].append(row[key])

    overall_results = {}
    for key in metrics_keys:
        if key != "name":
            overall_results[key] = sum(map(float, metrics[key])) / len(metrics[key])

    combined_csv_path = os.path.join(save_path, "combined_metrics.csv")
    with open(combined_csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_keys)
        writer.writeheader()
        for i in range(len(metrics["name"])):
            row = {key: metrics[key][i] for key in metrics_keys}
            writer.writerow(row)

    combined_json_path = os.path.join(save_path, "combined_metrics.json")
    with open(combined_json_path, "w") as f:
        json.dump(overall_results, f, indent=4)
