from functools import partial

from safe_gpu.safe_gpu import GPUOwner
owner = GPUOwner()

import argparse
import torch
import os

from transformers import BertForTokenClassification, BertTokenizerFast
from torch.utils.data import random_split
from datetime import datetime

from trainer import Trainer
from dataset import AlignmentDataset
import model


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-bert", action="store_true", default=False, help="Whether to train BERT as well. Note that this extremely increases training time.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--grad", type=int, default=10, help="Max grad norm")

    parser.add_argument("--model-path", help="Path to a model checkpoint.", default=None)
    parser.add_argument("--bert-path", default="bert-base-multilingual-uncased", help="Path to a pretrained BERT model. This is NOT used if --model-path is specified.")
    parser.add_argument("--tokenizer-path", default="bert-base-multilingual-uncased", help="Path to a tokenizer.")
    parser.add_argument("--save-path", help="Path to a directory where checkpoints are stored.")

    parser.add_argument("--ocr-path", help="Path to folder containing ocr.")
    parser.add_argument("--train-path", help="Path to a text file with training data.")
    parser.add_argument("--val-path", help="Path to a text file with validation data.")
    parser.add_argument("--test-path", help="Path to a text file with test data.")

    parser.add_argument("--labels", type=int, default=33, help="Number of labels, using this should be avoided.")

    args = parser.parse_args()
    return args


def load_dataset(data_path, ocr_path, batch_size, tokenizer, num_workers=0):
    dataset = AlignmentDataset(data_path=data_path, ocr_path=ocr_path, tokenizer=tokenizer)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader


if __name__ == "__main__":
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # model = BertForTokenClassification.from_pretrained(args.modelpath, num_labels=args.labels)
    model = model.build_model(num_labels=args.labels, model_path=args.model_path, pretrained_bert_path=args.bert_path)
    model = model.to(device)

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)

    load_data = partial(load_dataset, ocr_path=args.ocr_path, batch_size=args.batch_size, tokenizer=tokenizer)
    train_dataset = load_data(args.train_path)
    val_dataset = load_data(args.val_path)
    # test_dataset = load_data(args.test_path)

    # test_dataset = HandAnnotatedDataset("./page-txts", "annotations.json", tokenizer)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)
    # print("Test dataset loaded and DataLoader created.")

    trainer_settings = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_grad_norm": args.grad,
        "num_labels": args.labels,
        "bert": args.train_bert,
        "output_folder": args.save_path
    }

    os.makedirs(trainer_settings["output_folder"], exist_ok=True)

    trainer = Trainer(settings=trainer_settings, model=model, tokenizer=tokenizer)
    print("Trainer created.")

    print("Training starts ...")
    trainer.train(train_dataset, val_dataset)
    print("Training finished.")

    tokenizer.save_vocabulary(trainer_settings["output_folder"])
    print("Tokenizer saved.")

    # trainer.evaluate(test_dataset)

