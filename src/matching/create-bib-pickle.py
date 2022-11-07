import os
import typing
import argparse
import collections
import pickle


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bib", help="Path to bib file.")
    parser.add_argument("--out", help="Output file")

    args = parser.parse_args()
    return args


def parse_line(line: str):
    fields = line.split()

    if len(fields) < 4:
        raise ValueError("Line too short")

    if not line.startswith("mzk"):
        raise ValueError("Invalid line")

    card_id = fields[0]
    fields = fields[1:]
    try:
        ind_of_L = fields.index('L')
    except ValueError:
        try:
            ind_of_L = fields.index("I")
        except ValueError:
            try:
                ind_of_L = fields.index("S")
            except ValueError:
                ind_of_L = fields.index("H")

    entry_type = fields[:ind_of_L]
    content = ' '.join(fields[ind_of_L+1:])

    return card_id, entry_type[0], content


def main():
    args = parse_arguments()

    all_data: typing.Dict[str, typing.Dict[str, str]] = collections.defaultdict(dict)

    with open(args.bib, "r") as bib_file:
        for line in bib_file:
            try:
                file_id, field_id, content = parse_line(line)
            except ValueError:
                continue

            all_data[file_id][field_id] = content


    print("Processing done")
    with open(args.out, 'wb') as f:
        pickle.dump(all_data, f)

    print("DONE")

    return 0


if __name__ == "__main__":
    exit(main())

