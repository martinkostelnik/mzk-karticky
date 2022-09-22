""" Should we squeeze alignments if they are following each other with one char space?"""

"""
    We load data as 
    {
        filename: ({anno_label: [(from, to), ...]}, {alig_label: [(from, to), ...]}),
        filename: ({anno_label: [(from, to), ...]}, {alig_label: [(from, to), ...]}),
        filename: ({anno_label: [(from, to), ...]}, {alig_label: [(from, to), ...]}),
        filename: ({anno_label: [(from, to), ...]}, {alig_label: [(from, to), ...]}),
        .
        .
        .
    }
"""

"""
    MODE 1:
        We want to know how well the alignment script aligns on found fields, we don't care how well it can find those fields
        This means we only calculate errors in labels present in both alignment and annotation

        This also means we want to generate example files with alignment next to annotation
        to see how well it performs

        In case there are more alignments to one annotated label, select the one with the lowest error

        Even if we have Author (for example) in both annotation and alignment and the error is 100 %, this error is NOT
        accounted in the results. Remember, we wish to know the character alignment.

    MODE 2:
        Calculate error on per character basis.

        Two masked strings are created, one for annotation and one for alignment. For example:
        Let's consider a mask {Nothing: 0, Author: 1, Title: 2}. Each character from the OCR text is then
        replaced using this mask.
        Then we could get masks that look like this:
        Annotation: 0000111111111111100000000000000000222222222222222222200000000000000
        Alignment:  0000111111111111111100000000000022222222222222222222222200000000000

        We now iterate over all the characters and compare them. If two characters aren't the same, an alignment error
        has occured (which means that union gets incremented, but the intersection does not).
"""


import os
import json
import argparse

from src.helper import LABELS
from sklearn.metrics import classification_report

class ResultManager:
    def __init__(self, filenames: list):
        self.total_intersection = 0
        self.total_union = 0

        self.label_intersections = {label: 0 for label in LABELS}
        self.label_unions = {label: 0 for label in LABELS}

        self.file_errors = {
            filename: 
                {
                    "total_intersection": 0,
                    "total_union": 0,
                    "label_intersections": {},
                    "label_unions": {},
                } for filename in filenames }

        self.example_files = {filename: [] for filename in filenames}

        self.filenames = filenames

        self.classification_report = None

    def increment(self, intersection: int, union: int, filename: str, label: str):
        self.total_intersection += intersection
        self.total_union += union

        self.label_intersections[label] += intersection
        self.label_unions[label] += union

        self.file_errors[filename]["total_intersection"] += intersection
        self.file_errors[filename]["total_union"] += union

        try:
            self.file_errors[filename]["label_intersections"][label] += intersection
            self.file_errors[filename]["label_unions"][label] += union
        except KeyError:
            self.file_errors[filename]["label_intersections"][label] = intersection
            self.file_errors[filename]["label_unions"][label] = union

    def update_examples(self, filename: str, label: str, intersection: int, union: int, anno: tuple, alig: tuple, ocr: str):
        anno_from, anno_to = anno
        alig_from, alig_to = alig

        anno_text = ocr[anno_from:anno_to]
        alig_text = ocr[alig_from:alig_to]
        error = 100 * (1 - (intersection / union))

        self.example_files[filename].append(f"Label:      {label}")
        self.example_files[filename].append(f"Annotation: {anno_from}-{anno_to} {repr(anno_text)}")
        self.example_files[filename].append(f"Alignment:  {alig_from}-{alig_to} {repr(alig_text)}")
        self.example_files[filename].append(f"Error:      {error:.2f} % ({intersection} / {union} matched)\n")

    def save_file_errors(self, output_path: str):
        file_errors_path = os.path.join(output_path, r"file-errors")
        os.makedirs(file_errors_path, exist_ok=True)

        for filename in self.filenames:
            with open(os.path.join(file_errors_path, filename), "w") as f:
                if self.file_errors[filename]["total_union"] == 0:
                    print("Alignment script did not match any labels correctly.", file=f)
                    continue

                total_error = 1 - (self.file_errors[filename]["total_intersection"] / self.file_errors[filename]["total_union"])

                print(f"Total error: {total_error:.4f}\n", file=f)

                for (label, intersection), union in zip(self.file_errors[filename]["label_intersections"].items(), self.file_errors[filename]["label_unions"].values()):
                    print(f"{label}: {(1 - (intersection/ union)):.4f}", file=f)

                print(file=f)

                for i, line in enumerate(self.example_files[filename]):
                    print(line[:-1] if i == len(self.example_files[filename]) - 1 else line, file=f)

    def print_errors(self):
        try:
            total_error = 1 - (self.total_intersection / self.total_union)
        except ZeroDivisionError:
            print("All alignment files empty. Cannot calculate any error.")

        print(f"Total error: {total_error:.4f}\n")

        print(f"Label           Error")
        for (label, intersection), union in zip(self.label_intersections.items(), self.label_unions.values()):
            try:
                label_error = 1 - (intersection / union)
            except ZeroDivisionError:
                print(f"{label:<14}  -1")
            else:
                print(f"{label:<14}  {(1-(intersection / union)):.4f}")

        if self.classification_report is not None:
            print(f"\n{self.classification_report}")


EMPTY_MASK = '0'

MASK = {
    "Author": '1',
    "Title": '2',
    "Original title": '3',
    "Publisher": '4',
    "Pages": '5',
    "Series": '6',
    "Edition": '7',
    "References": '8',
    "ID": '9',
    "ISBN": 'a',
    "ISSN": 'b',
    "Topic": 'c',
    "Subtitle": 'd',
    "Date": 'e',
    "Institute": 'f',
    "Volume": 'g',
    }

INV_MASK = {val: key for key, val in MASK.items()}
INV_MASK["0"] = "O"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json", help="Path to JSON file with annotated data.", required=True)
    parser.add_argument("--alig", help="Path to folder with alignments.", required=True)
    parser.add_argument("--ocr", help="Path to folder with ocr.", required=True)

    parser.add_argument("--mode", type=int, default=1, help="Error calculation mode.", required=True)

    parser.add_argument("--output", help="Path to folder where output will be saved.")

    args = parser.parse_args()

    return args


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def create_mask(annotation: dict, ocr: str) -> str:
    parts = []
    ocr_copy = ocr

    for label, values in annotation.items():
        for value in values:
            start, end, with_  = value[0], value[1], MASK[label]
            parts.append([start, end, with_])

    parts.sort(key=lambda x: x[0])
    at = 0

    for part in parts:
        gap = part[0] - at
        l = part[1] - part[0]

        ocr_copy = ocr_copy[:at] + (gap * EMPTY_MASK) + (l * part[2]) + ocr_copy[part[1]:]
        at = part[1]

    return ocr_copy[:at] + (EMPTY_MASK * (len(ocr) - at))


def load_data(annotation_path: str, alignment_path: str) -> list:
    res = {}

    json_annotations = load_json(annotation_path)

    for annotated_file_json in json_annotations:
        if "label" not in annotated_file_json:
            continue

        annotation = {label: [] for label in LABELS}
        alignment = {label: [] for label in LABELS}

        for annotated_field_json in annotated_file_json["label"]:
            label = annotated_field_json["labels"][0]
            from_ = annotated_field_json["start"]
            to = annotated_field_json["end"]
            
            annotation[label].append((from_, to))

        filename = annotated_file_json["text"].rpartition("/")[2]

        for root, _, walk_filename in os.walk(alignment_path):
            with open(os.path.join(alignment_path, filename), "r") as f:
                for line in f:
                    s = line.split("\t")
                    label = s[0]
                    from_ = int(s[2])
                    to = int(s[3])

                    if label == "Original_title":
                        label = "Original title"

                    alignment[label].append((from_, to))

        res[filename] = (annotation, alignment)

    return res


def run_matching_mode(data: dict, result_manager: ResultManager, ocr_path: str):
    # For each file
    for filename, (file_annotations, file_alignments) in data.items():
        with open(os.path.join(ocr_path, filename), "r") as f:
            ocr = f.read()

        # For each label in given hand annotated file
        for label, items in file_annotations.items():

            # For each annotation of given label
            for anno_from, anno_to in items:
                lowest_error = 1
                intersection = 0
                union = 0
                res_alig_from = 0
                res_alig_to = 0

                # For each alignment of given label
                for alig_from, alig_to in file_alignments[label]:
                    alig_intersection = max(min(anno_to, alig_to) - max(anno_from, alig_from), 0)
                    alig_union = max(anno_to, alig_to) - min(anno_from, alig_from)

                    error = 1 - (alig_intersection / alig_union)

                    if error < lowest_error:
                        lowest_error = error
                        intersection = alig_intersection
                        union = alig_union
                        res_alig_from = alig_from
                        res_alig_to = alig_to

                # We found some matching alignment, update results
                if lowest_error != 1:
                    result_manager.increment(intersection, union, filename, label)
                    result_manager.update_examples(filename, label, intersection, union, (anno_from, anno_to), (alig_from, alig_to), ocr)


def run_masked_mode(data: dict, result_manager: ResultManager, ocr_path: str):
    all_anno = []
    all_alig = []

    for filename, (file_annotations, file_alignments) in data.items():
        with open(os.path.join(ocr_path, filename), "r") as f:
            ocr = f.read()

            masked_annotation = create_mask(file_annotations, ocr)
            masked_alignment = create_mask(file_alignments, ocr)

            all_anno.extend(list(masked_annotation))
            all_alig.extend(list(masked_alignment))
            
            for anno_char, alig_char in zip(masked_annotation, masked_alignment):
                if anno_char == alig_char and alig_char != EMPTY_MASK:
                    label = INV_MASK[anno_char]
                    result_manager.increment(1, 1, filename, label)

                if alig_char != anno_char:
                    label = INV_MASK[anno_char] if anno_char != EMPTY_MASK else INV_MASK[alig_char]
                    result_manager.increment(0, 1, filename, label)

    all_anno = [INV_MASK[c] for c in all_anno]
    all_alig = [INV_MASK[c] for c in all_alig]

    result_manager.classification_report = classification_report(all_anno, all_alig, zero_division=0)


if __name__ == "__main__":
    args = parse_arguments()

    data = load_data(annotation_path=args.json, alignment_path=args.alig)
    
    result = ResultManager(data.keys())
 
    if args.mode == 1:
        os.makedirs(args.output, exist_ok=True)
        run_matching_mode(data=data, result_manager=result, ocr_path=args.ocr)
        result.save_file_errors(args.output)
    elif args.mode == 2:
        run_masked_mode(data=data, result_manager=result, ocr_path=args.ocr)

    result.print_errors()
