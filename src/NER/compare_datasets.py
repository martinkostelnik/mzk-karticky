""" This script is used to compare two datasets.
    
    It takes two files containing file alignments and only
    calculates error on those files, which are present in both files.

    It uses Pytorch dataset to more handily get the OCR of the files.

    The comparison of two files is done on a character basis, where
    the characters from OCR are replaced by a mask of their label.
"""


__author__ = "Martin Kostelník"


import typing
import argparse

from dataset import AlignmentDataset
from sklearn.metrics import classification_report

from src.NER import helper


EMPTY_MASK = '0'

MASK = {
    "Author": '1',
    "Title": '2',
    "Original title": '3',
    "Publisher": '4',
    "Pages": '5',
    "Series": '6',
    "Edition": '7',
    "References": '8',
    "ID": '9',
    "ISBN": 'a',
    "ISSN": 'b',
    "Topic": 'c',
    "Subtitle": 'd',
    "Date": 'e',
    "Institute": 'f',
    "Volume": 'g',
    }

INV_MASK = {val: key for key, val in MASK.items()}
INV_MASK["0"] = "O"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inference", help="Path to a inference file.", required=True)
    parser.add_argument("--truth", help="Path to a file with ground truth.", required=True)
    parser.add_argument("--ocr", help="Path to LMDB with OCR.", required=True)
    parser.add_argument("--model", help="Path to folder with model config that generated the inference.", required=True)

    args = parser.parse_args()
    return args


def create_mask(annotation, ocr: str) -> str:
    parts = []
    ocr_copy = ocr

    for alignment in annotation.alignments:
        start, end, with_  = alignment.start, alignment.end, MASK[alignment.label]
        parts.append([start, end, with_])

    parts.sort(key=lambda x: x[0])
    at = 0

    for part in parts:
        gap = part[0] - at
        l = part[1] - part[0]

        ocr_copy = ocr_copy[:at] + (gap * EMPTY_MASK) + (l * part[2]) + ocr_copy[part[1]:]
        at = part[1]

    return ocr_copy[:at] + (EMPTY_MASK * (len(ocr) - at))


def main() -> int:
    args = parse_arguments()

    model_config = helper.ModelConfig.load(args.model)
    print("Model config loaded.")
    
    tokenizer = helper.build_tokenizer(args.model, model_config)
    print("Tokenizer loaded.")

    truth_dataset = AlignmentDataset(args.truth,
                                     args.ocr,
                                     tokenizer=tokenizer,
                                     model_config=model_config,
                                     min_aligned=0)

    print(f"Truth dataset loaded. Len = {len(truth_dataset)}.")

    inferred_dataset = AlignmentDataset(args.inference,
                                        args.ocr,
                                        tokenizer=tokenizer,
                                        model_config=model_config,
                                        min_aligned=0)

    print(f"Inferred dataset loaded. Len = {len(inferred_dataset)}.")

    truth_dict = {}
    for truth_dato in truth_dataset.data:
        truth_dict[truth_dato.file_id] = truth_dato
    print(f"Truth dataset dict created")

    truth_labels_all = []
    inferred_labels_all = []
    files_compared = 0
    wrong = 0

    print("Starting comparing ...")
    for inferred_dato in inferred_dataset.data:
        try:
            truth_dato = truth_dict[inferred_dato.file_id]
        except KeyError:
            continue
    
        truth_mask = create_mask(truth_dato, truth_dato.text)
        inferred_mask = create_mask(inferred_dato, inferred_dato.text)

        if len(truth_mask) != len(inferred_mask):
            wrong += 1
            continue

        truth_mask = [INV_MASK[l] for l in truth_mask]
        inferred_mask = [INV_MASK[l] for l in inferred_mask]

        truth_labels_all.extend(truth_mask)
        inferred_labels_all.extend(inferred_mask)
        files_compared += 1

        if not files_compared % 1000:
            print(f"{files_compared} / {len(inferred_dataset)} files compared.")

    print(classification_report(truth_labels_all, inferred_labels_all, zero_division=0, digits=4))
    print(f"\n{files_compared} files compared.")
    print(f"{wrong} files wrong")


if __name__ == "__main__":
    exit(main())
