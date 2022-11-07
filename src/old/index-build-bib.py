import argparse
import os
import pickle

import whoosh.index
from whoosh.fields import ID, TEXT


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--bib-pickle', required=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    with open(args.bib_pickle, 'rb') as f:
        bib_records = pickle.load(f)

    os.makedirs(args.index_dir, exist_ok=True)

    # TODO: Change schema to hold database information
    schema = whoosh.fields.Schema(
        record_id=ID(stored=True),
        content=TEXT,
    )

    index = whoosh.index.create_in(args.index_dir, schema)
    writer = index.writer()

    for r_id, r_content in bib_records.items():
        content = ' '.join(r_content.values())
        writer.add_document(record_id=r_id, content=content)\

    writer.commit()


if __name__ == '__main__':
    exit(main())
