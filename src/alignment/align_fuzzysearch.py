import argparse
import os
import typing

from fuzzysearch import find_near_matches
from src import helper



def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db", help="Path to folder with clean database records.")
    parser.add_argument("--ocr", help="Path to folder with transcription files.")
    parser.add_argument("--mapping", help="Path to file with mapping.")
    parser.add_argument("--output", help="Path to output folder.")

    parser.add_argument("--threshold", type=float, default=0.3, help="Alignment error threshold.")

    args = parser.parse_args()
    return args


def load_db_records(path: str) -> dict:
    data = {}

    with open(path) as file:
        for line in file:
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


def process_file(ocr_path: str, db_path: str, output_path: str, threshold: float):
    db_record = load_db_records(db_path)

    with open(ocr_path, "r") as f:
        ocr = f.read()

    output_path = os.path.join(output_path, ocr_path.rpartition("/")[2])

    with open(output_path, "w") as f:
        for label, record in db_record.items():
            max_dist = int(len(record) * threshold)
            matches = find_near_matches(record, ocr, max_l_dist=max_dist)
            
            lowest = min(matches, key=lambda x: x.dist, default=None)
            
            if lowest:
                print(f"{label}\t{repr(lowest.matched)}\t{lowest.start}\t{lowest.end}", file=f)


if __name__ == "__main__":
    args = parse_arguments()

    os.makedirs(args.output, exist_ok=True)

    mapping = helper.create_mapping(args.mapping)
    
    for ocr_filename, db_filename in mapping.items():
        # To process 200 hand annotated data, uncomment next line
        ocr_filename = ocr_filename.rpartition("/")[2]

        ocr_path = os.path.join(args.ocr, f"{ocr_filename}.gif.xml.txt")
        db_path = os.path.join(args.db, f"{db_filename}.txt")

        process_file(ocr_path, db_path, args.output, args.threshold)
