import argparse
import os
import typing
import collections

import whoosh.index
from whoosh.fields import ID, TEXT

from src import helper

from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--index-dir', required=True, help="Output folder in which the index will be stored.")
    parser.add_argument('--bib-file', required=True, help="mzk-bib file containing DB records.")
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    os.makedirs(args.index_dir, exist_ok=True)
    print("Output directory created.")
    
    normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
    print("Normalizer created.")

    schema = whoosh.fields.Schema(
        record_id=ID(stored=True),
        author=TEXT,
        title=TEXT,
        original_title=TEXT,
        publisher=TEXT,
        pages=TEXT,
        series=TEXT,
        edition=TEXT,
        references=TEXT,
        id=TEXT,
        isbn=TEXT,
        issn=TEXT,
        topic=TEXT,
        subtitle=TEXT,
        date=TEXT,
        institute=TEXT,
        volume=TEXT,
    )
    print("Whoosh schema created.")

    index = whoosh.index.create_in(args.index_dir, schema)
    writer = index.writer()
    print("Whoosh index and writer created.")

    entries_added = 0

    print("Filling index ...")
    with open(args.bib_file, "r") as bib_file:
        complete_record = {}
        prev_file_id = ""

        for i, line in enumerate(bib_file):
            try:
                file_id, field_id, field_val = helper.parse_line(line)
            except ValueError:
                continue

            if not i: # First entry
                prev_file_id = file_id

            records = helper.generate_db_records(field_id, field_val)

            if file_id != prev_file_id:
                writer.add_document(record_id=file_id, **complete_record)
                prev_file_id = file_id
                complete_record = {}
                
                entries_added += 1
                if not entries_added % 10000:
                    print(f"{entries_added} entries added to index.")
            
            for label, vals in records.items():
                label_lc = label.lower()
                for val in vals:
                    try:
                        complete_record[label_lc] += f" {val}"
                    except KeyError:
                        complete_record[label_lc] = f"{val}"

                complete_record[label_lc] = normalizer.normalize_str(complete_record[label_lc]).strip()

    writer.add_document(record_id=file_id, **complete_record) # Add last entry
    print(f"{entries_added + 1} entries added to index.")
    writer.commit()


if __name__ == '__main__':
    exit(main())