import argparse
import json

from src import helper


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json", help="Path to the json file containing annotations.")
    parser.add_argument("--mapping", help="Path to a mapping file.")
    parser.add_argument("--output", help="Output path.")

    args = parser.parse_args()
    return args


def load_annotations(path):
    with open(path, "r") as f:
        return json.load(f)


def fix_original_title(lines):
    return [line.replace("Original title", "Original_title") for line in lines]


def save_output(lines, output):
    with open(output, "w") as f:
        for line in lines:
            print(line, file=f)


if __name__ == "__main__":
    args = parse_arguments()

    filenames = helper.create_mapping(args.mapping).keys()
    annotations = load_annotations(args.json)

    entries = []

    for annotation in annotations:
        if "label" not in annotation:
            continue

        filename = annotation["text"].rpartition("/")[2].partition(".")[0]
        for tmp in filenames:
            if tmp.endswith(filename):
                entry = f"{tmp}.gif.xml.txt"
                break

        for annotated_item in annotation["label"]:
            entry += f'\t{annotated_item["labels"][0]} {annotated_item["start"]} {annotated_item["end"]}'

        entries.append(entry)

    entries = fix_original_title(entries)

    save_output(entries, args.output)
