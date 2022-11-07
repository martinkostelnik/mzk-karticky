import argparse
import os
import typing
import collections

import whoosh.index
from whoosh.fields import ID, TEXT

from src import helper


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--bib-file', required=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    print("Loading mzk-bib file ...")
    bib_records = helper.get_db_dict(args.bib_file)
    print("mzk-bib file loaded.")

    print(len(bib_records))

    os.makedirs(args.index_dir, exist_ok=True)
    print("Output directory created.")

    schema = whoosh.fields.Schema(
        record_id=ID(stored=True),
        author1=TEXT,
        author2=TEXT,
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

    for file_id, file_dict in bib_records.items():
        parsed_records = {}

        for field_id, field_val in file_dict.items():
            record = helper.generate_db_records(field_id, field_val)

            if "Author" in record:
                if "author2" not in parsed_records:
                    k = "author2" if "author1" in parsed_records else "author1"
                    record[k] = record["Author"]
                del record["Author"]

            parsed_records = {**parsed_records, **record}

        parsed_records = {key.lower(): val.strip() for key, val in parsed_records.items()}
        writer.add_document(record_id=file_id, **parsed_records)

    writer.commit()


if __name__ == '__main__':
    exit(main())
