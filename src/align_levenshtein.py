import os
import argparse
import itertools
import numpy as np
from collections import defaultdict

from pero_ocr.document_ocr.layout import TextLine
from pero_ocr.sequence_alignment import levenshtein_distance, levenshtein_alignment_substring

import helper

from timeout import timeout, TimeoutError


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-record", help="Path to folder with clean database records.")
    parser.add_argument("--transcription", help="Path to folder with transcription files.")
    parser.add_argument("--mapping", help="Path to file with mapping.")
    parser.add_argument("--output", help="Path to output folder.")
    parser.add_argument("--threshold", help="CER threshold for final filtering.", type=float, default=0.7)
    parser.add_argument("--max-candidates", help="Maximum lines to consider as a field", type=int, default=5)

    args = parser.parse_args()
    return args


def load_db_records(path):
    data = {}

    with open(path) as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                db_key, db_record = line.split(maxsplit=1)
                records = helper.generate_db_records(db_key, db_record)
                data = {**data, **records}

    # Remove empty records
    for key in list(data.keys()):
        if data[key].strip() == "" or len(data[key].strip()) <= 0:
            del data[key]

    return data


def load_transcription(path):
    lines = []
    with open(path, "r") as f:
        for line in f:
            if len(line) > 2:
                lines.append(TextLine(transcription=line[:-1])) # Omit "\n"

    return lines


def merge_db_records(data):
    text = ""
    boundaries = []

    for key in data:
        text += f"{data[key].strip()} "
        boundaries.append(len(text) - 1)

    text = text.strip()

    return text, boundaries


def save_mapping(db_mapping, output_path, transcription_path):
    with open(transcription_path, "r") as transcription_file:
        ocr = transcription_file.read()
    
    sep = chr(255)
    with open(output_path, "w") as file:
        for db_key in db_mapping:
            for line in db_mapping[db_key]:
                index = ocr.find(line.transcription)
                file.write(f"{sep}{db_key}{sep} {sep}{line.transcription}{sep} {sep}{index}{sep} {sep}{index + len(line.transcription)}{sep}\n")


def cer(hyp, ref, case_sensitive=False):
    if not case_sensitive:
        hyp = hyp.lower()
        ref = ref.lower()

    return levenshtein_distance(list(hyp), list(ref)) / len(ref)


def is_candidate(alignment):
    i = 0
    for db_char, line_char in alignment:
        if line_char is not None and db_char is not None and db_char.lower() == line_char.lower():
            i += 1

        if i >= 2:
            return True

    return False


def find_candidates(db_records, line):
    candidates = []

    db_text, boundaries = merge_db_records(db_records)
    #alignment = levenshtein_alignment(list(db_text), list(line.transcription))
    alignment = levenshtein_alignment_substring(list(db_text), list(line.transcription))
    for key, start, end in zip(db_records, [0] + boundaries[:-1], boundaries):
        if is_candidate(alignment[start:end]):
            candidates.append(key)

    return candidates


def evaluate(candidate_keys, db_records, line):
    cers = []

    for candidate_key in candidate_keys:
        candidate = db_records[candidate_key]
        cers.append(cer(line.transcription, candidate))

    return cers


# To find problematic files, a TimeoutError is raised after 60 seconds
# https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
@timeout(60)
def filter_and_sort_lines(db_record, lines):
    indices = list(range(len(lines)))

    best_cer = 1.0
    best_combination = []
    
    for length in range(1, len(indices) + 1):
        for subset in itertools.combinations(indices, length):
            for permutation in itertools.permutations(subset):
                text = ' '.join([lines[index].transcription for index in permutation])
                text_cer = cer(text, db_record)

                if text_cer < best_cer:
                    best_cer = text_cer
                    best_combination = permutation

    lines = [lines[index] for index in best_combination]

    return lines


# def title_matched(mapping):
#     for key in mapping:
#         if helper.title_pattern.matches(key, "") and len(mapping[key]) > 0:
#             return True

#     return False


def process_file(db_path, transcription_path, output_path, threshold, max_candidates):
    db_records = load_db_records(db_path)
    lines = load_transcription(transcription_path)
    db_records_mapping = defaultdict(list)

    for line in lines:
        candidates = find_candidates(db_records, line)

        if len(candidates) > 0:
            cers = evaluate(candidates, db_records, line)

            min_index = np.argmin(cers)
            if (len(db_records_mapping[candidates[min_index]])) < max_candidates:
                db_records_mapping[candidates[min_index]].append(line)

    for key in db_records_mapping:
        # The exception handling is here to find problematic files, to be removed later (maybe)
        try:
            lines = filter_and_sort_lines(db_records[key], db_records_mapping[key])
        except TimeoutError:
            print(f"Timeout reached on {transcription_path.rpartition('/')[2]}")
            return

        text_cer = cer(' '.join([line.transcription for line in lines]), db_records[key])

        if text_cer < threshold:
            db_records_mapping[key] = lines
        else:
            db_records_mapping[key] = []

    save_mapping(db_records_mapping, output_path, transcription_path)


def main():
    args = parse_arguments()

    mapping = helper.create_mapping(args.mapping)

    for ocr, id in mapping.items():
        ocr_filename = os.path.join(args.transcription, f"{ocr.rpartition('/')[2]}.gif.xml.txt")
        db_filename = os.path.join(args.db_record, f"{id}.txt")
        out_filename = os.path.join(args.output, f"{ocr.rpartition('/')[2]}.gif.xml.txt")

        print(f"Processing db entry {db_filename} with ocr {ocr_filename}")

        try:
            process_file(db_filename, ocr_filename, out_filename, args.threshold, args.max_candidates)
        except FileNotFoundError:
            continue

    return 0


if __name__ == "__main__":
    exit(main())
