import argparse

from model import NerBert
from trainer import Trainer
from dataset import prepare_training_data


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Set training mode, please provide ocr and alig.")
    parser.add_argument("--ocr", help="Path to folder containing ocr.")
    parser.add_argument("--alig", help="Path to folder containing alignments.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.train:
        data = prepare_training_data(args.ocr, args.alig)

        trainer_settings = {
            "train_ratio": 0.7,
            "val_ratio" : 0.15,
            "batch_size": 4,
            "epochs": 15,
            "learning_rate": 0.001,
            "max_grad_norm": 10,
        }

        model = NerBert()

        trainer = Trainer(data=data, settings=trainer_settings, model=model)

        trainer.train()
        trainer.evaluate()
    
    else:
        pass
