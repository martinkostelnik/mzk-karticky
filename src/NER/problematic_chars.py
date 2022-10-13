""" This script was used to file which characters are problematic when used together 
    with huggingface BertTokenizerFast. It requires an import of charset from PERO OCR.
"""

import typing
import argparse

import helper
from charset import all


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer-path", help="Path to a tokenizer.", required=True)

    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_arguments()

    tokenizer = helper.build_tokenizer(args.tokenizer_path)
    print("Tokenizer loaded.")

    unknowns = []
    truncated = []
    wrong = []

    for c in all:
        seq = [c]

        encoding = tokenizer(seq, is_split_into_words=True)
        words = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        reconstructed_c = words[1]

        if reconstructed_c == "[SEP]":
            truncated.append(c)
        elif reconstructed_c == "[UNK]":
            unknowns.append(c)  
        elif c.lower() != reconstructed_c:
            wrong.append(c)

    print(f"TRUNCATED ({len(truncated)}): {truncated}\n")
    print(f"UNKNOWN ({len(unknowns)}): {unknowns}\n")
    print(f"MISMATCHED ({len(wrong)}): {wrong}\n")
    print(f"Total: {len(truncated) + len(unknowns) + len(wrong)}")


if __name__ == "__main__":
    exit(main())
