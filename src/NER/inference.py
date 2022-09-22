import os
import typing
import argparse
import numpy as np

from model import build_model
from helper import IDS2LABELS, MAX_TOKENS_LEN, NUM_LABELS, calculate_confidence
from typing import Callable

import torch
from transformers import BertTokenizerFast


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", help="Path to a directory containing data for inference.")
    parser.add_argument("--model-path", help="Path to a trained model.")
    parser.add_argument("--tokenizer-path", help="Path to a tokenizer.")
    parser.add_argument("--save-path", help="Path to a output directory.")
    
    parser.add_argument("--threshold", default=0.0, type=float, help="Threshold for selecting field with enough model confidence.")
    parser.add_argument("--aggfunc", default="prod", type=str, help="Confidence aggregation function.")

    args = parser.parse_args()
    return args


def get_aggfunc(func_name: str) -> Callable:
    if func_name == "prod":
        return np.prod

    if func_name == "mean":
        return np.mean

    if func_name == "max":
        return np.max

    if func_name == "min":
        return np.min

    if func_name == "median":
        return np.median


def get_ocr(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


def logits_to_preds(tokenizer, logits, ids, offset_mapping) -> list:
    active_logits = logits.view(-1, NUM_LABELS)
    flattened_predictions = torch.argmax(active_logits, axis=1)

    wordpieces = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    wordpieces_preds = [IDS2LABELS[pred] for pred in flattened_predictions.cpu().numpy()]

    result = []

    for pred, mapping in zip(wordpieces_preds, offset_mapping.squeeze().tolist()):
        if mapping[0] == 0 and mapping[1] != 0:
            result.append(pred)

    return result


def infer(ocr: str, tokenizer, model) -> list:
    words = ocr.split()

    encoding = tokenizer(words,
                         padding="max_length",
                         is_split_into_words=True,
                         return_offsets_mapping=True,
                         truncation=True,
                         max_length=MAX_TOKENS_LEN,
                         return_tensors="pt")

    device = model.get_device()

    ids = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(ids, attention_mask=mask)[1][0]

    preds = logits_to_preds(tokenizer, logits, ids, encoding["offset_mapping"])
    confidence = calculate_confidence(logits)[1:len(preds) + 1].tolist()

    return list(zip(words, preds, confidence))


def find_offsets(text: str, data: list) -> list:
    result = []

    char_index = 0
    token_index = 0

    while token_index < len(data):
        token, label, conf = data[token_index]
        label = label if label == "O" else label[2:]

        text_slice = text[char_index:char_index+len(token)]

        if text_slice == token:
            result.append((token, label, char_index, char_index + len(token), conf))
            token_index += 1
            char_index += len(token)
        else:
            char_index += 1

    return result


def concat_token_offsets(offsets: list) -> list:
    result = []

    current_label = ""

    for _, label, from_, to, confidence in offsets:
        if label != current_label:
            result.append((label, from_, to, [confidence]))
            current_label = label
        else:
            _, current_from, _, confidences = result[-1]
            confidences.append(confidence)
            result[-1] = (label, current_from, to, confidences)

    return result


def aggregate_confidence(alignments: list, aggfunc: Callable) -> list:
    result = []

    for label, from_, to, confidences in alignments:
        result.append((label, from_, to, aggfunc(confidences).item()))

    return result


def save_result(path: str, result: list, ocr: str) -> None:
    output_folder = path.rpartition("/")[0]
    os.makedirs(output_folder, exist_ok=True)

    with open(path, "w") as f:
        for label, from_, to, _ in result:
            print(f"{label}\t{repr(ocr[from_:to])}\t{from_}\t{to}", file=f)


def save_output_dataset(data: list, path: str) -> None:
    with open(os.path.join(path, "dataset.all"), "w") as f:
        for line in data:
            print(line, file=f)


def main() -> int:
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_path=args.model_path)
    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)

    output_dataset = []

    print(f"Aggregation function: {args.aggfunc}")
    print(f"Confidence threshold: {args.threshold}")
    aggfunc = get_aggfunc(args.aggfunc)
    total = 0

    for root, _, filenames in os.walk(args.data_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            line = f"{file_path}"

            ocr = get_ocr(file_path)
            
            result = infer(ocr, tokenizer, model)

            token_offsets = find_offsets(ocr, result)
            alignments = concat_token_offsets(token_offsets)
            alignments = [alignment for alignment in alignments if alignment[0] != "O"]
        
            alignments = aggregate_confidence(alignments, aggfunc)
            alignments = [alignment for alignment in alignments if alignment[3] >= args.threshold]

            for label, from_, to, _ in alignments:
                line += f"\t{label} {from_} {to}"
                total += 1

            output_dataset.append(line)

            save_result(os.path.join(args.save_path, filename), alignments, ocr)

    save_output_dataset(output_dataset, args.save_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
