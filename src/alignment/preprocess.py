import os
import typing
import argparse
from collections import defaultdict

from src.helper import create_mapping


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mapping", help="Path to file with mapping.")
    parser.add_argument("--db-files", help="Path to DB files (mzk01.bib and mzk03.bib).", nargs="+")
    parser.add_argument("--output-dir", help="Path to output directory.")

    args = parser.parse_args()
    return args


def load_db_files(files, db_ids):
    data = {}
    for file in files:
        file_data = load_db_file(file, db_ids)

        for key in file_data:
            if key in data:
                data[key].append(file_data[key])
            else:
                data[key] = file_data[key]

    return data


def load_db_file(path, db_ids):
    data = defaultdict(list)

    with open(path) as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                try:
                    db_id, *_ = line.split(maxsplit=1)
                except ValueError:
                    continue

                if db_id in db_ids:
                    data[db_id].append(line)

    return data


def save_db_files(data, output_dir):
    for key in data:
        path = os.path.join(output_dir, f"{key}.txt")
        with open(path, "w") as file:
            for line in data[key]:
                file.write(f"{line}\n")


def clean_db_records(data:dict) -> dict:
    clean_data = defaultdict(list)

    for key in data:
        for line in data[key]:
            # TODO: Je dobry napad brat pouze 10:13? Ty pripadne dalsi dve hodnoty take muzou neco znamenat ...
            clean_data[key].append(f"{line[10:13]} {line[18:]}")

    return clean_data


def main() -> int:
    args = parse_arguments()

    mapping = create_mapping(args.mapping)
    print(f"Mapping loaded ({len(mapping)}).")

    db_ids = set(mapping.values())
    data = load_db_files(args.db_files, db_ids)
    print(f"DB files loaded ({len(data)}).")

    data = clean_db_records(data)
    print(f"Data cleaned.")

    save_db_files(data, args.output_dir)
    print(f"Data saved.")

    return 0


if __name__ == "__main__":
    exit(main())
