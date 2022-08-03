import argparse

from transformers import BertForTokenClassification, BertTokenizerFast

from model import NerBert
from trainer import Trainer
from dataset import prepare_training_data


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Set training mode, please provide ocr and alig.")
    parser.add_argument("--ocr", help="Path to folder containing ocr.")
    parser.add_argument("--alig", help="Path to folder containing alignments.")

    parser.add_argument("--load", action="store_true", help="Load model and tokenizer, please provide paths.")
    parser.add_argument("--model", help="Path to existing model.")
    parser.add_argument("--token", help="Path to existing tokenizer")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.load:
        pass
    else:
        model = BertForTokenClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=33)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")

    if args.train:
        data = prepare_training_data(args.ocr, args.alig)

        trainer_settings = {
            "train_ratio": 0.9,
            "val_ratio" : 0.05,
            "batch_size": 4,
            "epochs": 100,
            "learning_rate": 0.01,
            "max_grad_norm": 10,
        }

        # model = NerBert()
        trainer = Trainer(data=data, settings=trainer_settings, model=model)
        trainer.train()
        trainer.evaluate()
    
    else:
        pass
