import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np

import whoosh
import whoosh.index
import whoosh.scoring
import whoosh.qparser

from matching_util import records_match_score


def search_phrase(searcher, phrase):
    items = phrase.split()
    terms = [whoosh.query.Term('content', item) for item in items]
    query = whoosh.query.Or(terms)

    return searcher.search(query, limit=5)


def compose_search_phrase(raw_text):
    joined = ' OR '.join(raw_text.split())

    return joined


def ocr_line_acceptable(line):
    if '�' in line:
        return False
    if len(line) < 3:
        return False

    return True


def main(args):
    with open(args.bib_pickle, 'rb') as f:
        bib_records = pickle.load(f)

    logging.info(f'reading index...')
    index = whoosh.index.open_dir(args.index_dir)

    nb_cards_searched = 0

    fns_to_process = [fn for fn in os.listdir(args.card_dir) if fn.endswith('.txt')]
    logging.info(f'Found {len(fns_to_process)} files to process...')

    with index.searcher(weighting=whoosh.scoring.BM25F) as searcher:
        t0 = time.time()
        try:
            for fn in fns_to_process:
                logging.info(f'working {fn}...')
                
                with open(f'{args.card_dir}/{fn}') as f:
                    content = f.readlines()
                content = [line for line in content if ocr_line_acceptable(line)]

                nb_cards_searched += 1

                try:
                    phrase = compose_search_phrase(''.join(content))

                    t_debug = time.time()
                    results = search_phrase(searcher, phrase)
                    search_dur = time.time() - t_debug
                    logging.info(f'Index search took {search_dur:.1f}s')

                    match_scores = [records_match_score(content, bib_records[r['record_id']])[0] for r in results]

                    if max(match_scores) >= args.min_matched_lines:
                        print(fn, results[np.argmax(match_scores)]['record_id'], max(match_scores))
                    else:
                        logging.info(f"{fn}: No matches")

                except Exception as e:
                    logging.warning(f'{fn}: {e}')
                    continue

        except KeyboardInterrupt:
            pass

        t1 = time.time()

    dur = t1-t0
    nb_records = nb_cards_searched
    logging.info(f'Took {dur:.1f} seconds to search {nb_records} records. {dur/nb_records:.2f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--card-dir', required=True)
    parser.add_argument('--bib-pickle', required=True)
    parser.add_argument('--min-matched-lines', type=int, default=4)

    parser.add_argument('--logging-level', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        default='WARNING')
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(args.logging_level)

    main(args)
