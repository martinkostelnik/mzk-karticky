import argparse
import os
import time
import lmdb
import json
import random

import numpy as np

import whoosh
import whoosh.index
import whoosh.scoring

from fuzzysearch import find_near_matches
from src import helper
from src.NER.helper import load_ocr
from src.alignment.align_fuzzysearch import correct_overlaps
from src.alignment.timeout import timeout, TimeoutError


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

    query = whoosh.query.Or(fuzzy_terms)
    return searcher.search(query, limit=None)


# def ocr_line_acceptable(line):
#     if '�' in line:
#         return False
#     if len(line) < 3:
#         return False
#     if len(line) > 20:
#         return False

#     return True


# We tried to match inferred fields to db entries. Now we have several databse-IDS which
# potentionally match to current ocr card.
# We will now reverse the process: take the database records and try to find them all in the ocr.
# The number of matched fields will be our matching score.
def match_candidate(ocr: str, db: dict) -> list:
    fields = []

    for raw_label, raw_text in db.items():
        parsed_db_item = helper.generate_db_records(raw_label, raw_text) # raw_label == gov code | raw_text == dollars

        for label, text in parsed_db_item.items():
            for item in text:
                if len(item) < 4:
                    continue

                max_dist = get_max_dist(item)
                matches = find_near_matches(item, ocr, max_l_dist=max_dist)

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


def save_results(results, path):
    with open(os.path.join(path, "matching.txt"), "w") as match_f, \
         open(os.path.join(path, "alignment.txt"), "w") as alig_f:
        for matching, alignment in results:
            print(matching, file=match_f, end="")
            print(alignment,file=alig_f, end="")


def process_file(line, args):
    file_path, *alignments = line.split("\t")

    print(f"Matching {file_path}")

    ocr_txn = lmdb.open(args.ocr_lmdb, readonly=True, lock=False).begin()
    bib_txn = lmdb.open(args.bib_lmdb, readonly=True, lock=False).begin()
    ocr = load_ocr(file_path.strip(), ocr_txn)

    parsed_alignments = parse_alignments(alignments, ocr)

    index = whoosh.index.open_dir(args.index_dir)
    with index.searcher(weighting=whoosh.scoring.BM25F) as searcher:
        # Find candidates
        try:
            candidates = search_phrase(searcher, parsed_alignments)
        except TimeoutError:
            print(f"Timeout reached on file {file_path}, skipping")
            return None, None

        candidate_dbs = []
        for candidate in candidates:
            bib_string = bib_txn.get(candidate["record_id"].encode()).decode()
            bib_json = json.loads(bib_string)
            candidate_dbs.append(bib_json)

        candidate_alignments = [match_candidate(ocr, candidate_db) for candidate_db in candidate_dbs] # [[alignments..], [alignments..], ..]
        match_scores = [len(candidate_alignment) for candidate_alignment in candidate_alignments]

        if not len(candidate_alignments):
            return None, None

        best_candidate_idx = np.argmax(match_scores)
        best_candidate_id = candidates[best_candidate_idx]['record_id']
        best_candidate_alignment = candidate_alignments[best_candidate_idx]
        best_candidate_aligned = len(best_candidate_alignment)

        if best_candidate_aligned >= args.min_matched_lines:
            print(f"Best match for file {file_path} is {best_candidate_id} with score: {best_candidate_aligned}")

            matching_output = f"{file_path}\t{best_candidate_id}\t{best_candidate_aligned}\n"
            
            alignment_output = f"{file_path}"
            for alignment in best_candidate_alignment:
                alignment_output += f"\t{alignment['label']} {alignment['from']} {alignment['to']}"
            alignment_output += "\n"

            return matching_output, alignment_output
        else:
            print(f"Not enough matches found for file {file_path} (must be higher than {args.min_matched_lines}, was {best_candidate_aligned})")
            return None, None


def main() -> int:
    args = parse_arguments()

    print("Reading inference data ...")
    with open(args.inference_path, "r") as f:
        inference_lines = f.readlines()
    print("Inference data read.")

    cards_searched = 0
    result = []

    print("Starting search ...")
    t0 = time.time()
        
    # DEBUG: SELECT RANDOM INDEX TO START FROM AND TAKE ONLY 10 
    idx = random.randint(0, 2e6)                             ##
    inference_lines = inference_lines[idx:idx + 10]          ##
    ###########################################################

    try:
        for line in inference_lines:
            matching, alignment = process_file(line, args)
            cards_searched += 1

            if matching and alignment:
                result.append((matching, alignment))
    except KeyboardInterrupt:
        pass

    t1 = time.time()
    dur = t1 - t0
    print(f'Took {dur:.1f} seconds to search {cards_searched} records. {dur / cards_searched:.2f}s')

    print("Saving results ...")
    save_results(result, args.out_path)
    print(f"Results saved to {args.out_path}")


if __name__ == '__main__':
    exit(main())
