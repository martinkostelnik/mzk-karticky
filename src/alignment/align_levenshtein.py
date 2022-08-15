import os
import argparse
import itertools
import numpy as np
from collections import defaultdict

from pero_ocr.document_ocr.layout import TextLine
from pero_ocr.sequence_alignment import levenshtein_distance, levenshtein_distance_substring, levenshtein_alignment_substring

from multiprocessing import Pool
from functools import partial

from src import helper

from timeout import timeout, TimeoutError


class Border:
    def __init__(self, start, end):
        if start > end:
            start, end = end, start

        self.start = start
        self.end = end

    def __str__(self):
        return f"{self.start}:{self.end}"

    def intersects(self, other_border):
        return self.start < other_border.start < self.end or self.start < other_border.end < self.end

    def is_inside(self, other_border):
        return  self.start < other_border.start < self.end and self.start < other_border.end < self.end


"""
threshold   error
0.05        0.941782
0.10        0.868518
0.15        0.794391
0.20        0.736693
0.25        0.699514
0.30        0.641934
0.35        0.590829
0.40        0.561588
0.45        0.521777
0.50        0.500310
0.55        0.468965
0.60        0.441028
0.65        0.419010
0.70        0.405674
0.75        0.398083
0.80        0.393863
0.85        0.385290
0.90        0.385755
0.95        0.385267
1.00        0.383286
"""
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-record", help="Path to folder with clean database records.")
    parser.add_argument("--transcription", help="Path to folder with transcription files.")
    parser.add_argument("--mapping", help="Path to file with mapping.")
    parser.add_argument("--output", help="Path to output folder.")
    parser.add_argument("--threshold", help="CER threshold for final filtering.", type=float, default=0.8)
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
                lines.append(TextLine(transcription=line.strip()))

    return lines


def merge_db_records(data):
    text = ""
    boundaries = []

    for key in data:
        text += f"{data[key].strip()} "
        boundaries.append(len(text) - 1)

    text = text.strip()

    return text, boundaries


def save_mapping(db_mapping, output_path, all_lines):
    text = ' '.join(line.transcription for line in all_lines)

    sep = '\t'

    # Comment this to align 200 hand annotated
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
   
    output_path = os.path.join(output_path.partition("/")[0], output_path.rpartition("/")[2])
    
    with open(output_path, "w") as file:
        for db_key in db_mapping:
            borders = db_mapping[db_key]
            for border in borders:
                if border is not None:
                    file.write(f"{db_key}{sep}{text[border.start:border.end]}{sep}{border.start}{sep}{border.end}\n")


def cer(hyp, ref, case_sensitive=False, substring=False):
    if not case_sensitive:
        hyp = hyp.lower()
        ref = ref.lower()

    if substring:
        f = levenshtein_distance_substring
    else:
        f = levenshtein_distance

    return f(list(hyp), list(ref)) / len(ref)


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


def find_substring(text, line):
    alignment = levenshtein_alignment_substring(list(text), list(line))
    for i in range(len(alignment)):
        if alignment[i][1] is not None:
            break

    for j in range(len(alignment)-1, -1, -1):
        if alignment[j][1] is not None:
            break

    return text[i:j+1]


def merge_consecutive_borders(borders):
    if len(borders) == 0:
        return borders

    new_borders = []
    start = borders[0].start
    end = borders[0].end

    for current_border, next_border in zip(borders[:-1], borders[1:]):
        if current_border.end + 1 == next_border.start:
            end = next_border.end
        else:
            new_borders.append(Border(start, end))
            start = next_border.start
            end = next_border.end

    new_borders.append(Border(start, end))
    return new_borders


def get_borders(all_lines, key_lines, db_record):
    all_text = ' '.join(line.transcription for line in all_lines)
    key_text = ' '.join(line.transcription for line in key_lines)

    alignment = levenshtein_alignment_substring(list(key_text), list(db_record))

    for first_line_start in range(len(alignment)):
        if alignment[first_line_start][1] is not None:
            break

    for last_line_end in range(len(alignment)-1, -1, -1):
        if alignment[last_line_end][1] is not None:
            break

    borders = []
    for i, line in enumerate(key_lines):
        line_start = all_text.index(line.transcription)
        line_end = line_start + len(line.transcription)

        if i == 0:
            line_start += first_line_start

        if i == len(key_lines) - 1:
            line_end = line_start + last_line_end + 1

        borders.append(Border(line_start, line_end))

    borders = merge_consecutive_borders(borders)
    return borders


def extract_text(all_lines, borders):
    all_text = ' '.join(line.transcription for line in all_lines)
    text_parts = [all_text[b.start:b.end].strip() for b in borders]
    return ' '.join(text_parts)


def refine_borders(left_border, right_border, spaces):
    inside_spaces = [space for space in spaces if left_border <= space <= right_border]

    if len(inside_spaces) == 1:
        return inside_spaces[0]

    elif len(inside_spaces) > 1:
        return inside_spaces[len(inside_spaces)//2]

    else:
        left_spaces = [space for space in spaces if space < left_border]
        right_spaces = [space for space in spaces if space > right_border]

        left_space = max(left_spaces) if len(left_spaces) > 0 else None
        right_space = min(right_spaces) if len(right_spaces) > 0 else None

        if left_space is None and right_space is None:
            return (left_border + right_border) // 2

        elif left_space is None:
            return right_space

        elif right_space is None:
            return left_space

        else:
            if abs(left_space - left_border) < abs(right_space - right_border):
                return left_space

            else:
                return right_space


def solve_overlapping_borders(db_records_mapping, lines):
    text = ' '.join([line.transcription for line in lines])
    spaces = [i for i, c in enumerate(text) if c == " "]

    for key1 in db_records_mapping:
        for key2 in db_records_mapping:
            if key1 != key2:
                for index1, border1 in enumerate(db_records_mapping[key1]):
                    if border1 is not None:
                        for index2, border2 in enumerate(db_records_mapping[key2]):
                            if border2 is not None:
                                if border1.intersects(border2):
                                    if border1.is_inside(border2):
                                        db_records_mapping[key1][index1] = None

                                    elif border2.is_inside(border1):
                                        db_records_mapping[key2][index2] = None

                                    elif border1.start < border2.start < border1.end:
                                        refined_border = refine_borders(border2.start, border1.end, spaces)
                                        border1.end = refined_border
                                        border2.start = refined_border

                                    else:
                                        refined_border = refine_borders(border1.start, border2.end, spaces)
                                        border1.start = refined_border
                                        border2.end = refined_border


def process_file(db_path, transcription_path, output_path, threshold, max_candidates):
    db_records = load_db_records(db_path)
    all_lines = load_transcription(transcription_path)
    db_records_mapping = defaultdict(list)

    for line in all_lines:
        candidates = find_candidates(db_records, line)

        if len(candidates) > 0:
            cers = evaluate(candidates, db_records, line)

            min_index = np.argmin(cers)
            db_records_mapping[candidates[min_index]].append(line)

    for key in db_records_mapping:
        if len(db_records_mapping[key]) > max_candidates:
            print(f"Too many lines for key '{key}' ({len(db_records_mapping[key])}).")
            db_records_mapping[key] = []
            continue

        # The exception handling is here to find problematic files, to be removed later (maybe)
        try:
            key_lines = filter_and_sort_lines(db_records[key], db_records_mapping[key])

        except TimeoutError:
            print(f"Timeout reached on {transcription_path}")
            db_records_mapping[key] = []
            continue

        borders = get_borders(all_lines, key_lines, db_records[key])
        text = extract_text(all_lines, borders)
        text_cer = cer(text, db_records[key])

        if text_cer < threshold:
            db_records_mapping[key] = borders
        else:
            db_records_mapping[key] = []

    solve_overlapping_borders(db_records_mapping, all_lines)
    save_mapping(db_records_mapping, output_path, all_lines)


def process_mapping_item(data, transcription, db_record, output, threshold, max_candidates):
    ocr, id = data
    
    # To create alignments for 200 hand annotated card
    ocr_filename = os.path.join(transcription, f"{ocr.rpartition('/')[2]}.gif.xml.txt")
    
    # ocr_filename = os.path.join(transcription, f"{ocr}.gif.xml.txt")
    db_filename = os.path.join(db_record, f"{id}.txt")
    out_filename = os.path.join(output, f"{ocr}.gif.xml.txt")

    print(f"Processing db entry {db_filename} with ocr {ocr_filename}")

    if os.path.isfile(out_filename):
        print(f"File '{out_filename}' already exists.")
        return 2

    try:
        process_file(db_filename, ocr_filename, out_filename, threshold, max_candidates)
    except FileNotFoundError:
        return 1

    return 0


def main():
    args = parse_arguments()

    os.makedirs(args.output, exist_ok=True)

    mapping = helper.create_mapping(args.mapping)
    processing_function = partial(process_mapping_item, transcription=args.transcription,
                                                        db_record=args.db_record,
                                                        output=args.output,
                                                        threshold=args.threshold,
                                                        max_candidates=args.max_candidates)

    pool = Pool(processes=4)
    pool.map(processing_function, mapping.items())

    return 0


if __name__ == "__main__":
    exit(main())
