import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignments-dir", help="Path to the root directory with alignments.")
    parser.add_argument("--output", help="Path to the output file.")

    args = parser.parse_args()
    return args


def process_file(path, file_id, data):
    annotations = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                field, _, start, end = line.split("\t")
                field = field.replace(" ", "_")
                annotations.append(f"{field} {start} {end}")

    if len(annotations) > 0:
        data[file_id] = '\t'.join(annotations)


def process_alignments(root_dir):
    data = {}

    for base_dir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(base_dir, file)
            file_id = file_path[len(root_dir) + 1:]
            process_file(file_path, file_id, data)

    return data


def save_alignments(data, path):
    with open(path, "w") as file:
        for key, value in data.items():
            file.write(f"{key}\t{value}\n")


def main():
    args = parse_arguments()

    data = process_alignments(args.alignments_dir)
    save_alignments(data, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
