import os
import typing
import argparse

from helper import create_mapping, clean_db_record


RAW_DB_FOLDER = r"db"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mapping", help="Path to file with mapping.")
    parser.add_argument("--db", help="Path to folder containing mzk01.bib and mzk03.bib")

    args = parser.parse_args()
    return args


def extract_raw_db(mapping: dict, db_path: str) -> None:
    os.makedirs(RAW_DB_FOLDER, exist_ok=True)

    p1 = os.path.join(db_path, "mzk01.bib")
    p2 = os.path.join(db_path, "mzk03.bib")

    for _, id in mapping.items():
        os.system(f"cat {p1} {p2} | grep {id} > {RAW_DB_FOLDER}/{id}.txt")


def clean_db_records(mapping:dict) -> None:
    for _, id in mapping.items():
        path = os.path.join(RAW_DB_FOLDER, f"{id}.txt")

        clean_db_record(path, path)


def main() -> int:
    args = parse_arguments()

    mapping = create_mapping(args.mapping)

    extract_raw_db(mapping, args.db)
    clean_db_records(mapping)

    return 0


if __name__ == "__main__":
    exit(main())
