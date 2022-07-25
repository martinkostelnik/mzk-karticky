import os
import typing
import argparse
import json

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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", help="Path to JSON file with annotated data.")
    parser.add_argument("--alignments", help="Path to folder with the alignments.")
    parser.add_argument("--ocr", help="Path to folder with ocr.")

    args = parser.parse_args()

    return args


def load_annotations(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_ocr(path: str) -> str:
    with open(path, "r") as file:
        return file.read()


def load_alignment(path:str) -> list:
    res = []

    with open(path, "r") as file:
        for line in file:
            s = line.split(chr(255))
            d = {"labels": [s[1]], "start": int(s[5]), "end": int(s[7])}
            res.append(d)

    return res


def create_mask(annotation: dict, ocr: str) -> str:
    parts = []
    ocr_copy = ocr

    for field in annotation:
        start, end, with_  = field["start"], field["end"], MASK[field["labels"][0]]
        parts.append([start, end, with_])

    parts.sort(key=lambda x: x[0])
    at = 0

    for part in parts:
        gap = part[0] - at
        l = part[1] - part[0]

        ocr_copy = ocr_copy[:at] + (gap * EMPTY_MASK) + (l * part[2]) + ocr_copy[part[1]:]
        at = part[1]

    return ocr_copy[:at] + (EMPTY_MASK * (len(ocr) - at))


def get_field_errors(intersections: dict, unions: dict) -> dict:
    field_errors = {label: 0.0 for label in MASK}

    for label in field_errors:
        try:
            field_errors[label] = 1 - intersections[label] / unions[label]
        except ZeroDivisionError:
            field_errors[label] = -1

    return field_errors


def save_file_errors(file_errors: list) -> None:
    with open("file_alignment_errors.json", "w") as f:
        f.write(json.dumps(file_errors, indent=4))


def main() -> int:
    args = parse_arguments()

    annotations = load_annotations(args.annotations)

    total_intersection = 0
    total_union = 0

    field_intersections = {label: 0 for label in MASK}
    field_unions = {label: 0 for label in MASK}

    file_errors = []

    for annotated_file in annotations:
        if "label" in annotated_file:# and annotated_file["text"] == "/data/local-files/?d=mzk_karticky/page-txts/01600556.gif.xml.txt": # Filter out wrong matches
            filename = annotated_file["text"].rpartition("/")[2]

            ocr = load_ocr(os.path.join(args.ocr, filename))
            alignment = load_alignment(os.path.join(args.alignments, filename))

            masked_annotation = create_mask(annotated_file["label"], ocr)
            masked_alignment = create_mask(alignment, ocr)

            file_intersections = {}
            file_unions = {}

            for anno_char, alig_char in zip(masked_annotation, masked_alignment):
                if alig_char == anno_char and alig_char != EMPTY_MASK:
                    total_intersection += 1
                    total_union += 1

                    field_intersections[INV_MASK[alig_char]] += 1
                    field_unions[INV_MASK[alig_char]] += 1

                    if INV_MASK[alig_char] in file_intersections:
                        file_intersections[INV_MASK[alig_char]] += 1
                        file_unions[INV_MASK[alig_char]] += 1
                    else:
                        file_intersections[INV_MASK[alig_char]] = 1
                        file_unions[INV_MASK[alig_char]] = 1

                if alig_char != anno_char:
                    wrong_label = INV_MASK[anno_char] if anno_char != EMPTY_MASK else INV_MASK[alig_char]

                    total_union += 1

                    field_unions[wrong_label] += 1

                    if wrong_label in file_unions:
                        file_unions[wrong_label] += 1
                    else:
                        file_unions[wrong_label] = 1
            
            file_intersections = {**file_intersections, **{label: 0 for label in file_unions if label not in file_intersections}}
            file_error = {label: 1 - file_intersections[label] / file_unions[label] for label in file_unions}
            file_error["Total error"] = 1.0 if not sum(file_unions.values()) else 1 - sum(file_intersections.values()) / sum(file_unions.values())
            file_error["Filename"] = filename
            file_errors.append(file_error)

    field_errors = get_field_errors(field_intersections, field_unions)

    print(f"Total error is: 1 - {total_intersection}/{total_union} = {1 - total_intersection / total_union}\n")
    print(f"{json.dumps(field_errors, indent=4)}")

    save_file_errors(file_errors)

    return 0


if __name__ == "__main__":
    main()
