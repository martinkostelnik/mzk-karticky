import os
import lmdb
import argparse

from src import helper

from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bib", help="Path to the bib file.")
    parser.add_argument("--out", help="Path to the output folder.")

    args = parser.parse_args()
    return args


def write(cur, key, value):
    cur.put(key.encode(), value.encode())


def load(path):
    with open(path) as file:
        text = file.read()

    return text


def generate_content(record: dict) -> str:
    content = ""
    for label, values in record.items():
        content += f"{label}"
        for value in values:
            content += f"\t{value}"
        content += "\n"

    return content


def main():
    args = parse_arguments()

    normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
    print("Normalizer created.")

    gb100 = 100000000000
    entries_added = 0

    env = lmdb.open(args.out, map_size=gb100)
    with env.begin(write=True) as txn:
        cur = txn.cursor()

        with open(args.bib, "r") as bib_file:
            prev_file_id = ""
            complete_record = {}

            for i, line in enumerate(bib_file):
                try:
                    file_id, field_id, field_val = helper.parse_line(line)
                except ValueError:
                    continue
                
                if not i: # First line
                    prev_file_id = file_id

                records = helper.generate_db_records(field_id, field_val) # dict

                # If file_id changes, we have the whole file ready, commit to db and reset
                if file_id != prev_file_id:
                    content = generate_content(complete_record)
                    content = normalizer.normalize_str(content)

                    write(cur, prev_file_id, content)
                    prev_file_id = file_id
                    complete_record = {}

                    entries_added += 1
                    if not entries_added % 10000:
                        print(f"{entries_added} files added to LMDB.")

                for label, values in records.items():
                    for value in values:
                        try:
                            complete_record[label].append(value)
                        except KeyError:
                            complete_record[label] = [value]

            content = generate_content(complete_record)
            content = normalizer.normalize_str(content)
            write(cur, file_id, content) # Write last file
            
            print(f"{entries_added + 1} files added to LMDB.")

    return 0


if __name__ == "__main__":
    exit(main())
