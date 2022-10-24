import typing
import helper
from dataset import AlignmentDataset
from sklearn.metrics import classification_report

INFERRED_DATASET_PATH = r"/home/xkoste12/mzk-karticky/test-inference-output/dataset.all"
INFERRED_DATASET_OCR_PATH = r"/home/xkoste12/mzk-karticky/data/page-txts"
TEST_DATASET_PATH = r"/home/xkoste12/mzk-karticky/data/alignment.test"
TEST_DATASET_OCR_PATH = r"/mnt/xkoste12/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-ocr"
CHECKPOINT_PATH = r"/home/xkoste12/mzk-karticky/experiments/2022-10-14/e12/checkpoints"


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


def create_mask(annotation, ocr: str) -> str:
    parts = []
    ocr_copy = ocr

    for alignment in annotation.alignments:
        start, end, with_  = alignment.start, alignment.end, MASK[alignment.label]
        parts.append([start, end, with_])

    parts.sort(key=lambda x: x[0])
    at = 0

    for part in parts:
        gap = part[0] - at
        l = part[1] - part[0]

        ocr_copy = ocr_copy[:at] + (gap * EMPTY_MASK) + (l * part[2]) + ocr_copy[part[1]:]
        at = part[1]

    return ocr_copy[:at] + (EMPTY_MASK * (len(ocr) - at))

def main() -> int:
    model_config = helper.ModelConfig.load(CHECKPOINT_PATH)
    print("Model config loaded.")
    
    tokenizer = helper.build_tokenizer(CHECKPOINT_PATH, model_config)
    print("Tokenizer loaded.")

    truth_dataset = AlignmentDataset(TEST_DATASET_PATH,
                                     TEST_DATASET_OCR_PATH,
                                     tokenizer=tokenizer,
                                     model_config=model_config,
                                     min_aligned=0)

    inferred_dataset = AlignmentDataset(INFERRED_DATASET_PATH,
                                        INFERRED_DATASET_OCR_PATH,
                                        tokenizer=tokenizer,
                                        model_config=model_config,
                                        min_aligned=0)

    truth_labels_all = []
    inferred_labels_all = []

    for inferred_dato in inferred_dataset.data:
        inferred_filename = inferred_dato.file_id.rpartition("/")[2]

        inferred_mask = create_mask(inferred_dato, inferred_dato.text)

        for truth_dato in truth_dataset.data:
            truth_filename = truth_dato.file_id.rpartition("/")[2]
            
            if truth_filename == inferred_filename:
                truth_mask = create_mask(truth_dato, truth_dato.text)
                break
        else:
            continue

        assert len(inferred_mask) == len(truth_mask)

        truth_mask = [INV_MASK[l] for l in truth_mask]
        inferred_mask = [INV_MASK[l] for l in inferred_mask]

        truth_labels_all.extend(truth_mask)
        inferred_labels_all.extend(inferred_mask)

    print(classification_report(truth_labels_all, inferred_labels_all, zero_division=0))


if __name__ == "__main__":
    exit(main())
