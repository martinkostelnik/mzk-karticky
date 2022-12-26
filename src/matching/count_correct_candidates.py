""" This script is used to count if and how many correct candidates are among the candidates
    for each OCR file when searching the index.
"""


import os
import typing
import argparse
from filter_good_cards import process_alig_file, process_match_file


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cards", required=True, help="File containing alignments of all cards, we need to filter these to get the good cards")
    parser.add_argument("--candidates", required=True, help="Folder containing parallel folders of candidates.")
    parser.add_argument("--old", required=True, help="Old matching file.") 
    parser.add_argument("--new", required=True, help="New matching file")
    
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_arguments()

    in_candidates = 0 # How many times the correct ID is in the candidates
    correctly_matched = 0 # How many times we matched correctly
    unmatched = 0 # How many times the well aligned card is not in the new matching
    total_filenames_candidates = 0

    # Load filenames of well aligned cards
    good_filenames = process_alig_file(args.cards)
    print(f"Good filenames loaded, len = {len(good_filenames)}")

    # Find DB IDs of well aligned cards according to old matching
    good_cards = {}
    with open(args.old, "r") as f:
        for line in f:
            s = line.split()
            filename = s[0].replace("-", "/").strip()
            if filename in good_filenames:
                good_cards[filename] = s[1].strip()
    print(f"IDS of well aligned cards loaded, len = {len(good_cards)}")

    # Load new matching as {filename: ID, ...}
    matching = {}
    with open(args.new, "r") as f:
        for line in f:
            s = line.split()
            filename = s[0].strip()
            matching[filename] = s[1].strip()[6:]
    print(f"New matching loaded, len = {len(matching)}")

    # Go by candidate files one by one
    for root, _, filenames in os.walk(args.candidates):
        for filename in filenames:
            if filename != "candidates.txt":
                continue

            with open(os.path.join(root, filename), "r") as f:
                for line in f:
                    s = line.split()
                    filename = s[0].strip()
                    
                    if filename not in good_cards.keys():
                        continue

                    candidate_list = s[1:]
                    candidate_list = [c[6:] for c in candidate_list] # correct mzk prefixes

                    correct_id = good_cards[filename]
                    total_filenames_candidates += 1

                    if correct_id in candidate_list:
                        in_candidates += 1

    # Let's count how many times the matching is correct
    for good_card_filename, good_card_id in good_cards.items():
        try:
            if good_card_id == matching[good_card_filename]:
                correctly_matched += 1
        except:
            unmatched += 1

    print(f"{in_candidates / total_filenames_candidates} ({in_candidates} / {total_filenames_candidates})")
    print(f"{correctly_matched / (len(good_cards) - unmatched)} ({correctly_matched} / ({len(good_cards)} - {unmatched}))")

    return 0


if __name__ == "__main__":
    exit(main())
