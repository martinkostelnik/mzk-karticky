import torch
import argparse
import typing

import helper

from tester import Tester
from model import build_model
from dataset import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", help="Path to a model checkpoint.", required=True)
    parser.add_argument("--config-path", help="Path to json containing model config.", required=True)
    parser.add_argument("--tokenizer-path", help="Path to a tokenizer.", required=True)
    parser.add_argument("--data-path", help="Path to a text file with test data.", required=True)
    parser.add_argument("--ocr-path", help="Path to a directory containing ocr.", required=True)

    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = helper.ModelConfig.load(args.config_path)
    print("Model config loaded.")

    tokenizer = helper.build_tokenizer(args.tokenizer_path, model_config)
    print("Tokenizer loaded.")

    model = build_model(tokenizer=tokenizer, model_path=args.model_path, model_config=model_config)
    model = model.to(device)
    model.eval()
    print("Model loaded.")
    
    BATCH_SIZE = 16
    data = load_dataset(args.data_path, args.ocr_path, BATCH_SIZE, tokenizer, model_config)
    print("Dataset loaded.")

    tester = Tester(model)
    print("Tester created.")
    
    tester.test(data)

    return 0


if __name__ == "__main__":
    exit(main())
