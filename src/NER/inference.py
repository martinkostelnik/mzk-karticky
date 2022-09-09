import os
import typing
import argparse

from model import build_model
from dataset import IDS2LABELS

import torch
from transformers import BertTokenizerFast


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", help="Path to a directory containing data for inference.")
    parser.add_argument("--model-path", help="Path to a trained model.")
    parser.add_argument("--tokenizer-path", help="Path to a tokenizer.")
    parser.add_argument("--save-path", help="Path to a output directory.")

    parser.add_argument("--labels", type=int, default=33, help="Number of labels, using this should be avoided.")
    parser.add_argument("--max-len", type=int, default=256, help="Max encoding length, using this should be avoided.")

    args = parser.parse_args()
    return args


def get_ocr(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


def logits_to_preds(tokenizer, logits, ids, offset_mapping, num_labels: int) -> list:
    active_logits = logits.view(-1, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)

    wordpieces = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    wordpieces_preds = [IDS2LABELS[pred] for pred in flattened_predictions.cpu().numpy()]

    result = []

    for pred, mapping in zip(wordpieces_preds, offset_mapping.squeeze().tolist()):
        if mapping[0] == 0 and mapping[1] != 0:
            result.append(pred)

    return result


def infer(ocr: str, tokenizer, model, max_len: int) -> list:
    encoding = tokenizer(ocr.split(),
                         padding="max_length",
                         is_split_into_words=True,
                         return_offsets_mapping=True,
                         truncation=True,
                         max_length=max_len,
                         return_tensors="pt")

    ids = encoding["input_ids"].to(model.get_device())
    mask = encoding["attention_mask"].to(model.get_device())

    with torch.no_grad():
        logits = model(ids, attention_mask=mask)[1][0]

    preds = logits_to_preds(tokenizer, logits, ids, encoding["offset_mapping"], model.num_labels)

    return list(zip(ocr.split(), preds))


def save_result(path: str, result: list) -> None:
    output_folder = path.rpartition("/")[0]
    os.makedirs(output_folder, exist_ok=True)

    # Remove empty tokens and store labels (preds without I/B-)
    filtered = [(token, pred[2:]) for token, pred in result if pred != "O"]

    fields = {label: "" for label in set([x[1] for x in filtered])}

    for token, label in filtered:
        fields[label] += f" {token}"

    with open(path, "w") as f:
        for key, val in fields.items():
            print(f"{key} {repr(val.strip())}", file=f)


def save_output_dataset(data: list, path: str) -> None:
    with open(os.path.join(path, "dataset.all"), "w") as f:
        for line in data:
            print(line, file=f)


def main() -> int:
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_labels=args.labels, model_path=args.model_path)
    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)

    output_dataset = []

    for root, _, filenames in os.walk(args.data_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            line = f"{filename}"
            
            result = infer(get_ocr(file_path), tokenizer, model, args.max_len)

            for token, pred in result:
                line += f"\t{token} {pred}"
            output_dataset.append(line)

            save_result(os.path.join(args.save_path, filename), result)

    save_output_dataset(output_dataset, args.save_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
