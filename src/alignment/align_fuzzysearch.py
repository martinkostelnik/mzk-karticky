import argparse
import os
import typing

from fuzzysearch import find_near_matches
from src import helper

from multiprocessing import Pool
from functools import partial


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db", help="Path to folder with clean database records.")
    parser.add_argument("--ocr", help="Path to folder with transcription files.")
    parser.add_argument("--mapping", help="Path to file with mapping.")
    parser.add_argument("--output", help="Path to output folder.")

    parser.add_argument("--threshold", type=float, default=0.6, help="Alignment error threshold.")

    args = parser.parse_args()
    return args


def load_db_records(path: str) -> dict:
    data = {}

    with open(f"{path}.txt", "r") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                db_key, db_record = line.split(maxsplit=1)
                records = helper.generate_db_records(db_key, db_record)
                data = {**data, **records}

    # Remove empty records
    for key in list(data.keys()):
        if data[key].strip() == "" or len(data[key].strip()) <= 0:
            del data[key]

    return data


def save_alignment(lines: list, output_path: str) -> None:
    os.makedirs(output_path.rpartition("/")[0], exist_ok=True)

    with open(f"{output_path}.gif.xml.txt", "w") as f:
        for line in lines:
            print(f"{line['label']}\t{repr(line['text'])}\t{line['from']}\t{line['to']}", file=f)


def correct_overlaps(lines: list) -> list:
    lines.sort(key=lambda x: x["from"])

    for index1, line1 in enumerate(lines):
        for index2, line2 in enumerate(lines):
            if index1 != index2 and line1 is not None and line2 is not None:
                if line1["from"] <= line2["from"] < line2["to"] <= line1["to"]:
                    lines[index2] = None

                elif line1["from"] < line2["from"] < line1["to"] < line2["to"]:
                    middle = (line2["from"] + line1["to"]) // 2
                    line1["to"] = middle
                    line2["from"] = middle + 1

    lines = [line for line in lines if line is not None]

    return lines


def process_file(data, output_path: str, threshold: float, ocr_folder: str, db_folder: str) -> None:
    ocr_path, db_path = data
    output_file_path = os.path.join(output_path, ocr_path)

    if os.path.isfile(f"{output_file_path}.gif.xml.txt"):
        print(f"File already exists {output_file_path}")
        return

    db_record = load_db_records(os.path.join(db_folder, db_path))

    path = ocr_path
    # Uncomment this to align 200 hand annotated
    # path = ocr_path.rpartition("/")[2]

    with open(f"{os.path.join(ocr_folder, path)}.gif.xml.txt", "r") as f:
        ocr = f.read()

    lines = []

    for label, record in db_record.items():
        max_dist = min(int(len(record) * threshold), 12)
        matches = find_near_matches(record, ocr, max_l_dist=max_dist)

        lowest = min(matches, key=lambda x: x.dist, default=None)

        if lowest:
            lines.append({"label": label, "text": lowest.matched, "from": lowest.start, "to": lowest.end})

    try:
        lines = correct_overlaps(lines)
    except:
        print(f"Error during correcting overlaps {output_file_path}")
        return

    for line in lines:
        line["text"] = ocr[line["from"]:line["to"]]

    save_alignment(lines, output_file_path)

    print(f"Saved {output_file_path}")


if __name__ == "__main__":
    args = parse_arguments()

    mapping = helper.create_mapping(args.mapping)
    print(f"Mapping loaded ({len(mapping)})")

    processing_function = partial(process_file, ocr_folder=args.ocr,
                                                output_path=args.output,
                                                threshold=args.threshold,
                                                db_folder=args.db)

    pool = Pool(processes=5)
    pool.map(processing_function, mapping.items())
