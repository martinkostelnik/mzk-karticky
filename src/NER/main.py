import argparse
import torch
import os

from transformers import BertForTokenClassification, BertTokenizerFast
from torch.utils.data import random_split
from datetime import datetime

from trainer import Trainer
from dataset import FullDataset, HandAnnotatedDataset


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--train", action="store_true", help="Run in training mode.")

    parser.add_argument("--bert", action="store_true", default=False, help="Whether to train BERT as well. Note that this extremely increases training time.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--grad", type=int, default=10, help="Max grad norm")
    parser.add_argument("--train-ratio", type=float, default=0.90, help="Ratio of training data.")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Ratio of validating data.")

    parser.add_argument("--load", action="store_true", help="Load model and tokenizer")
    parser.add_argument("--modelpath", default="bert-base-multilingual-uncased", help="Path to model and tokenizer")

    parser.add_argument("--ocr", help="Path to folder containing ocr.")
    parser.add_argument("--alig", help="Path to folder containing alignments.")

    parser.add_argument("--labels", type=int, default=33, help="Number of labels, using this should be avoided.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    model = BertForTokenClassification.from_pretrained(args.modelpath, num_labels=args.labels)
    tokenizer = BertTokenizerFast.from_pretrained(args.modelpath)

    if args.train:
        data = FullDataset(args.ocr, args.alig, tokenizer, args.debug)

        train_size = int(args.train_ratio * len(data))
        val_size = len(data) - train_size

        train_set, val_set = random_split(data, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch, shuffle=True, num_workers=2)

        test_dataset = HandAnnotatedDataset("./page-txts", "annotations.json", tokenizer)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

        trainer_settings = {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "max_grad_norm": args.grad,
            "num_labels": args.labels,
            "bert": args.bert,
            "output_folder": f"model/{int(datetime.timestamp(datetime.now()))}",
            "debug": args.debug,
        }

        os.makedirs(trainer_settings["output_folder"])

        trainer = Trainer(settings=trainer_settings, model=model, tokenizer=tokenizer)

        try:
            trainer.train(train_loader, val_loader)
        finally:
            tokenizer.save_vocabulary(trainer_settings["output_folder"])
            trainer.evaluate(test_loader)
    
    else: # Inference mode
        pass
