import torch
import argparse
import typing

from transformers import BertTokenizerFast

from tester import Tester
from model import build_model
from helper import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", help="Path to a model checkpoint.")
    parser.add_argument("--tokenizer-path", help="Path to a tokenizer.")
    parser.add_argument("--data-path", help="Path to a text file with test data.")
    parser.add_argument("--ocr-path", help="Path to a directory containing ocr.")

    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_path=args.model_path)
    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)

    data = load_dataset(args.data_path, args.ocr_path, 16, tokenizer)

    tester = Tester(model)
    tester.test(data)

    return 0


if __name__ == "__main__":
    exit(main())
