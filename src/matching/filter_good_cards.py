""" This script is used to filter such cards that are well aligned.
"""


import os
import argparse
import typing


MIN_ALIGNED = 4
MUST_ALIGN = set(["Author", "Title", "ID"])


def parse_arguments():
    parser = argparse.ArgumentParser()

    # I used all 3 datasets for some reason
    # It might be easier to do `cat train val test > all` in further use 
    parser.add_argument("--train", required=True, help="Train alignments")
    parser.add_argument("--val", required=True, help="Validation alignments")
    parser.add_argument("--test", required=True, help="Test alignments")

    parser.add_argument("--old", required=True, help="Old matching file.")
    parser.add_argument("--new", required=True, help="New matching file")
    
    args = parser.parse_args()

    return args


def process_alig_file(path: str) -> list:
    result = []

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        filename, *alignments = line.split("\t")
        labels = []

        for alignment in alignments:
            label, *_ = alignment.split(" ")
            labels.append(label)

        labels = set(labels)

        if MUST_ALIGN.issubset(labels) and len(labels) >= MIN_ALIGNED:
            result.append(filename)

    return result


def process_match_file(path:str) -> dict:
    result = []

    with open(path, "r") as f:
        for i, line in enumerate(f):
            unpacked = line.split()
            filename = unpacked[0].replace("-", "/")
            id = unpacked[1]

            result.append((filename, id))

    return result


def main() -> int:
    args = parse_arguments()

    train_filenames = process_alig_file(args.train)
    val_filenames = process_alig_file(args.val)
    test_filenames = []

    with open(args.test, "r") as f:
        for line in f:
            filename, *_ = line.split("\t")
            test_filenames.append(filename)

    print(f"Found {len(train_filenames)} good training samples.")
    print(f"Found {len(val_filenames)} good validation samples.")
    print(f"Found {len(test_filenames)} good testing samples.")

    all_filenames = set(train_filenames + val_filenames + test_filenames)

    print(f"Found {len(all_filenames)} good samples.\n")

    old_matching = set([(key, id) for key, id in process_match_file(args.old) if key in all_filenames])
    new_matching = set([(key, id[6:]) for key, id in process_match_file(args.new) if key in all_filenames])

    print(f"Found {len(old_matching)} ids of good samples in old matching.")
    print(f"Found {len(new_matching)} ids of good samples in new matching.\n")

    intersection_ = old_matching.intersection(new_matching)
    print(f"Found {len(intersection_)} matching ids of good samples.")

    for filename, id in list(intersection_):
        print(f"{filename}\t{id}")

    return 0


if __name__ == "__main__":
    exit(main())
