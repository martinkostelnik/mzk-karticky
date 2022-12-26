""" This script is used to compare two files containing card matching. These are both in the same format:

        <file_id>\t<db_id>\t<n_mathed>\n
        .
        .
        .

    Output are two numbers:
        1. Number of cards that have the same match.
        2. Number of cards that don't have any match in old matching ilfe
"""


import os
import argparse
import typing


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--new', required=True, help="Path to new matching file.")
    parser.add_argument('--old', required=True, help="Path to old matching file.")

    args = parser.parse_args()
    return args


def load_file(path: str) -> dict:
    result = {}

    with open(path, "r") as f:
        for line in f:
            if len(line) <= 0 or line == "":
                continue

            key, id, _ = line.split()
            key = key.replace("-", "/")
            result[key] = id

    return result

def main() -> int:
    args = parse_arguments()

    old = load_file(args.old)
    new = load_file(args.new)

    matched = 0
    different_card = 0

    print(len(old))
    print(len(new))

    for key, id in new.items():
        try:
            # WARNING: Watch out fo the 6: offset
            if id[6:] == old[key]:
                matched += 1
        except KeyError:
            different_card += 1
            continue

    print(matched)
    print(different_card)
    return 0


if __name__ == "__main__":
    exit(main())
    