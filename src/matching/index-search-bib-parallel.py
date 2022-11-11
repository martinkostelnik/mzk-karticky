import argparse
import os
import sys
import time
import lmdb
import traceback
import psutil
import json

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

from multiprocessing import Pool
from functools import partial

        
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--index-dir', required=True, help="Path to a directory containing bib index.")
    parser.add_argument('--ocr-lmdb', required=True, help="Path to lmdb directory.")
    parser.add_argument('--min-matched-lines', type=int, default=3, help="How many fields must match to consider it a correct match.")
    parser.add_argument("--inference-path", required=True, help="Path to a file containing inferred data in dataset format.")
    parser.add_argument('--bib-lmdb', required=True, help="Path to a bib file.")
    parser.add_argument("--out-path", required=True, help="Output directory.")

    args = parser.parse_args()
    return args


def get_max_dist(text, threshold=0.25, limit=6):
    if len(text) < 6:
        return 1

    if len(text) < 11:
        return 2

    return min(int(len(text) * threshold), limit)


@timeout(30)
def search_phrase(searcher, alignments):
    fuzzy_terms = []

    for label, texts in alignments.items():
        for text in texts:
            max_dist = get_max_dist(text)

            words = [word for word in text.split() if len(word) > 3]

            for word in words:
                if label == "Author":
                    fuzzy_terms.append(whoosh.query.FuzzyTerm("author1", word, maxdist=2, prefixlength=0))
                    fuzzy_terms.append(whoosh.query.FuzzyTerm("author2", word, maxdist=2, prefixlength=0))
                else:
                    fuzzy_terms.append(whoosh.query.FuzzyTerm(label.lower(), word, maxdist=2, prefixlength=0))

    # lines = ocr.split("\n")
    # line_terms = []
    # for line in lines:
    #     line = line.strip()

    #     if ocr_line_acceptable(line):
    #         max_dist = get_max_dist(line)
    #         line_terms.append(whoosh.query.FuzzyTerm("content"), line, maxdist=line, prefixlength=0)

    # query = whoosh.query.Or(fuzzy_terms + line_terms)

    query = whoosh.query.Or(fuzzy_terms)
    # print(query)
    return searcher.search(query, limit=None) # TODO: Increase limit?


def ocr_line_acceptable(line):
    if '�' in line:
        return False
    if len(line) < 3:
        return False
    if len(line) > 20:
        return False

    return True


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
        try:
            result[label].append(ocr[int(start):int(end)])
        except KeyError:
            result[label] = [ocr[int(start):int(end)]]

    return result


# def save_results(alignment, matching, path):
#     with open(os.path.join(path, "alignment.txt"), "w") as f:
#         f.write(alignment)

#     with open(os.path.join(path, "matching.txt"), "w") as f:
#         f.write(matching)

def save_results(result, path):
    matching_output = ""
    alignment_output = ""

    for value in result:
        if not value:
            continue

        matching_line, alignment_line = value
        matching_output += matching_line
        alignment_output += alignment_line

    with open(os.path.join(path, "alignment.txt"), "w") as f:
        f.write(alignment_output)

    with open(os.path.join(path, "matching.txt"), "w") as f:
        f.write(matching_output)


def process_file(line, args):
    # process = psutil.Process(os.getpid())
    # print(f"Process {process} currently using {process.memory_info().rss / 1024 ** 2} MB")  # in bytes
    file_path, *alignments = line.split("\t")

    print(f"\nMatching {file_path}")

    ocr_txn = lmdb.open(args.ocr_lmdb, readonly=True, lock=False).begin()
    bib_txn = lmdb.open(args.bib_lmdb, readonly=True, lock=False).begin()
    ocr = load_ocr(file_path.strip(), ocr_txn)

    parsed_alignments = parse_alignments(alignments, ocr)

    index = whoosh.index.open_dir(args.index_dir)
    with index.searcher(weighting=whoosh.scoring.BM25F) as searcher:
        try:
            results = search_phrase(searcher, parsed_alignments)
        except TimeoutError:
            print(f"Timeout reached on file {file_path}, skipping")
            return None

        records = []
        for r in results:
            records.append(json.loads(bib_txn.get(r["record_id"].encode()).decode()))

        matches = [match_candidate(ocr, r) for r in records]
        match_scores = [len(match) for match in matches]
        print(f"Found {len(matches)} candidates.")

        if not len(matches):
            return None, None

        if max(match_scores) >= args.min_matched_lines:
            print(f"Best match for file {file_path} is {results[np.argmax(match_scores)]['record_id']} with score: {max(match_scores)}")

            matching_output = f"{file_path}\t{results[np.argmax(match_scores)]['record_id']}\t{max(match_scores)}\n"
            
            alignment_output = f"{file_path}"
            for f_line in matches[np.argmax(match_scores)]:
                alignment_output += f"\t{f_line['label']} {f_line['from']} {f_line['to']}"
            alignment_output += "\n"

            return matching_output, alignment_output
        else:
            print(f"Not enough matches found for file {file_path} (must be higher than {args.min_matched_lines}")
            return None, None


def main():
    args = parse_arguments()

    # print("Loading mzk-bib file ...")
    # bib_records = helper.get_db_dict(args.bib_file)
    # print("mzk-bib file loaded.")

    print("Reading inference data ...")
    with open(args.inference_path, "r") as f:
        inference_lines = f.readlines()
    print("Inference data read.")
    
    process = psutil.Process(os.getpid())
    print(f"Before search, {process.memory_info().rss / 1024 ** 2} MB of memory is used.")  # in bytes

    print("Starting search ...")

    t0 = time.time()
    processing_function = partial(process_file, args=args)
    print("Partial function created")

    pool = Pool(processes=4)
    print("Pool created")

    result = []

    try:
        for match, alig in pool.imap_unordered(processing_function, inference_lines):
            if match and alig:
                result.append((match, alig))
    except KeyboardInterrupt:
        pass

    t1 = time.time()

    dur = t1 - t0
    # nb_records = nb_cards_searched
    # print(f'Took {dur:.1f} seconds to search {nb_records} records. {dur / nb_records:.2f}s')
    print(f"Took {dur:.1f} seconds to search the records.")

    print("Saving results ...")
    # save_results(alignment_output, matching_output, args.out_path)
    save_results(result, args.out_path)
    print(f"Results saved to {args.out_path}")


if __name__ == '__main__':
    exit(main())
