import argparse
import os
import sys
import time
import lmdb
import traceback

import numpy as np

import whoosh
import whoosh.index
import whoosh.scoring
import whoosh.qparser

from fuzzysearch import find_near_matches
from src import helper
from src.NER.helper import load_ocr
from src.alignment.align_fuzzysearch import correct_overlaps
from src.alignment.timeout import timeout, TimeoutError


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--lmdb-path', required=True)
    parser.add_argument('--min-matched-lines', type=int, default=3)
    parser.add_argument("--inference-path", required=True)
    parser.add_argument('--bib-file', required=True)
    parser.add_argument("--out-path", required=True)

    args = parser.parse_args()
    return args


def get_max_dist(text, threshold=0.25, limit=6):
    if len(text) < 6:
        return 1

    if len(text) < 11:
        return 2

    return min(int(len(text) * threshold), limit)

@timeout(60)
def search_phrase(searcher, alignments):
    fuzzy_terms = []

    for label, text in alignments.items():
        max_dist = get_max_dist(text)

        if label == "Author":
            fuzzy_terms.append(whoosh.query.FuzzyTerm("author1", text, maxdist=max_dist, prefixlength=0))
            fuzzy_terms.append(whoosh.query.FuzzyTerm("author2", text, maxdist=max_dist, prefixlength=0))
        else:
            fuzzy_terms.append(whoosh.query.FuzzyTerm(label.lower(), text, maxdist=max_dist, prefixlength=0))

    query = whoosh.query.Or(fuzzy_terms)
    print(query)
    return searcher.search(query, limit=10) # TODO: Increase limit?


# def ocr_line_acceptable(line):
#     if '�' in line:
#         return False
#     if len(line) < 3:
#         return False

#     return True


# We tried to match inferred fields to db entries. Now we have several databse-IDS which
# potentionally match to current ocr card.
# We will now reverse the process: take the database records and try to find them all in the ocr.
# The number of matched fields will be our matching score.
def match_candidate(ocr: str, db_entries: dict) -> list:
    fields = []

    for raw_label, raw_text in db_entries.items():
        record = helper.generate_db_records(raw_label, raw_text)

        for label, text in record.items():
            max_dist = get_max_dist(text)

            try:
                matches = find_near_matches(text, ocr, max_l_dist=max_dist)
            except ValueError:
                continue

            lowest = min(matches, key=lambda x: x.dist, default=None)

            if lowest:
                fields.append({"label": label, "text": lowest.matched, "from": lowest.start, "to": lowest.end})

    try:
        fields = correct_overlaps(fields)
    except:
        print(f"Error during correcting overlaps")
        return []

    for line in fields:
        line["text"] = ocr[line["from"]:line["to"]]

    return fields


def parse_alignments(alignments, ocr):
    result = {}

    for alignment in alignments:
        label, start, end = alignment.split()
        result[label] = ocr[int(start):int(end)]

    return result


def save_results(alignment, matching, path):
    with open(os.path.join(path, "alignment.txt"), "w") as f:
        f.write(alignment)

    with open(os.path.join(path, "matching.txt"), "w") as f:
        f.write(matching)


def main():
    args = parse_arguments()

    print("Reading index ...")
    index = whoosh.index.open_dir(args.index_dir)
    print("Index loaded.")

    print("Loading mzk-bib file ...")
    bib_records = helper.get_db_dict(args.bib_file)
    print("mzk-bib file loaded.")

    print("Opening LMDB")
    txn = lmdb.open(args.lmdb_path, readonly=True, lock=False).begin()
    print("LMDB open.")

    nb_cards_searched = 0
    matching_output = ""
    alignment_output = ""

    print("Starting search ...")
    with index.searcher(weighting=whoosh.scoring.BM25F) as searcher:
        t0 = time.time()

        try:
            with open(args.inference_path, "r") as f:
                for line in f:
                    print()
                    file_path, *alignments = line.split("\t")

                    print(f"Matching {file_path}")

                    ocr = load_ocr(file_path, txn)
                    parsed_alignments = parse_alignments(alignments, ocr)

                    nb_cards_searched += 1

                    t_debug = time.time()

                    try:
                        results = search_phrase(searcher, parsed_alignments)
                    except TimeoutError:
                        print(f"Timeout reached on file {file_path}, skipping")
                        continue

                    search_dur = time.time() - t_debug
                    print(f'Index search took {search_dur:.1f}s')

                    matches = [match_candidate(ocr, bib_records[r["record_id"]]) for r in results]
                    match_scores = [len(match) for match in matches]
                    print(f"Found {len(matches)} candidates with scores: {match_scores}.")

                    if not len(matches):
                        continue

                    if max(match_scores) >= args.min_matched_lines:
                        print(f"Best match for file {file_path} is {results[np.argmax(match_scores)]['record_id']} with score: {max(match_scores)}")

                        matching_output += f"{file_path}\t{results[np.argmax(match_scores)]['record_id']}\t{max(match_scores)}\n"
                       
                        alignment_output += f"{file_path}"
                        for f_line in matches[np.argmax(match_scores)]:
                            alignment_output += f"\t{f_line['label']} {f_line['from']} {f_line['to']}"
                        alignment_output += "\n"
                    else:
                        print(f"Not enough matches found for file {file_path} (must be higher than {args.min_matched_lines}")

        except KeyboardInterrupt:
            pass

        t1 = time.time()

    dur = t1 - t0
    nb_records = nb_cards_searched
    print(f'Took {dur:.1f} seconds to search {nb_records} records. {dur / nb_records:.2f}s')

    print("Saving results ...")
    save_results(alignment_output, matching_output, args.out_path)
    print(f"Results saved to {args.out_path}")


if __name__ == '__main__':
    exit(main())
