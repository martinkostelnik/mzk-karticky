import typing
import os
import argparse
import lmdb

from helper import load_ocr
from src.helper import LABEL_TRANSLATIONS
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Path to a file containing dataset.", required=True)
    parser.add_argument("--ocr", help="Path to a OCR or LMDB folder.", required=True)
    parser.add_argument("--out", help="Path to an output folder.", required=True)

    args = parser.parse_args()
    return args


def save_file(path: str, result: str) -> None:
    with open(path, "w") as f:
        f.write(result)


def main() -> int:
    args = parse_arguments()

    txn = lmdb.open(args.ocr, readonly=True, lock=False).begin() if "lmdb" in args.ocr else None

    with open(args.dataset, "r") as f:
        dataset = f.readlines()

    os.makedirs(args.out, exist_ok=True)

    for line in dataset:
        output = ""

        key, *alignments = line.split("\t")

        ocr = load_ocr(key.strip(), txn)

        for alignment in alignments:
            label, from_, to = alignment.split(" ")
            field_text = ocr[int(from_):int(to)].replace("\n", " ")
            output += f"{LABEL_TRANSLATIONS[label]}: {field_text}\n"
        
        save_file(os.path.join(args.out, key.replace("/", "-")), output)

    return 0


if __name__ == "__main__":
    exit(main())
