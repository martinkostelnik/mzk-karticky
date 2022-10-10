import os
import typing
import argparse
import numpy as np
import torch

import helper

from model import build_model


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", help="Path to a directory containing data for inference.", required=True)
    parser.add_argument("--config-path", help="Path to a json file contatining model config.", required=True)
    parser.add_argument("--model-path", help="Path to a trained model.", required=True)
    parser.add_argument("--tokenizer-path", help="Path to a tokenizer.", required=True)
    parser.add_argument("--save-path", help="Path to a output directory.", required=True)
    
    parser.add_argument("--threshold", default=0.0, type=float, help="Threshold for selecting field with enough model confidence.")
    parser.add_argument("--aggfunc", default="prod", type=str, help="Confidence aggregation function.")

    args = parser.parse_args()
    return args


def get_aggfunc(func_name: str) -> typing.Callable:
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


def logits_to_preds(model_config, tokenizer, logits, ids, offset_mapping) -> list:
    active_logits = logits.view(-1, model_config.num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)
    
    wordpieces = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    wordpieces_preds = [model_config.ids2labels[pred] for pred in flattened_predictions.cpu().numpy()]

    result = []

    for tok, pred, mapping in zip(wordpieces, wordpieces_preds, offset_mapping.squeeze().tolist()):
        if mapping[0] == 0 and mapping[1] != 0 and tok != helper.LINE_SEPARATOR:
            result.append(pred)

    return result


def infer(ocr: str, tokenizer, model, filename) -> list:
    words = ocr.replace("\n", f" {helper.LINE_SEPARATOR} ").split() if model.config.sep else ocr.split()

    encoding = tokenizer(words,
                         padding="max_length",
                         is_split_into_words=True,
                         return_offsets_mapping=True,
                         truncation=True,
                         max_length=model.config.max_sequence_len,
                         return_tensors="pt")

    device = model.get_device()

    ids = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(ids, attention_mask=mask)[1][0]

    preds = logits_to_preds(model.config, tokenizer, logits, ids, encoding["offset_mapping"])

    confidence = helper.calculate_confidence(logits)[1:len(preds) + 1].tolist()

    if sum(encoding["attention_mask"].squeeze().tolist()) != 256:
        assert len(ocr.split()) == len(preds), f"{filename}"    
        assert len(ocr.split()) == len(confidence)
    else:
        print(f"File {filename} is being truncated due to maximum bert input length.")

    return list(zip(ocr.split(), preds, confidence))


def find_offsets(text: str, data: list, format: list) -> list:
    result = []

    char_index = 0
    token_index = 0

    while token_index < len(data):
        token, label, conf = data[token_index]

        if format == ["I", "O", "B"]:
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


def aggregate_confidence(alignments: list, aggfunc: typing.Callable) -> list:
    result = []

    for label, from_, to, confidences in alignments:
        result.append((label, from_, to, aggfunc(confidences).item()))

    return result


def save_result(path: str, result: list, ocr: str) -> None:
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
    print(f"Device: {device}")

    model_config = helper.ModelConfig.load(args.config_path)
    print("Model config loaded.")

    tokenizer = helper.build_tokenizer(args.tokenizer_path, model_config)
    print("Tokenizer loaded.")

    model = build_model(tokenizer=tokenizer, model_path=args.model_path, model_config=model_config)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    output_dataset = []

    aggfunc = get_aggfunc(args.aggfunc)
    print(f"Aggregation function: {args.aggfunc}")
    print(f"Confidence threshold: {args.threshold}")

    os.makedirs(args.save_path, exist_ok=True)
    print("Output directory created.")

    print("Inference starting ...")
    for root, _, filenames in os.walk(args.data_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            line = f"{file_path}"

            ocr = get_ocr(file_path)
            result = infer(ocr, tokenizer, model, filename)

            token_offsets = find_offsets(ocr, result, model_config.format)
            alignments = concat_token_offsets(token_offsets)
            alignments = [alignment for alignment in alignments if alignment[0] != "O"]
        
            # alignments = aggregate_confidence(alignments, aggfunc)
            # alignments = [alignment for alignment in alignments if alignment[3] >= args.threshold]

            for label, from_, to, _ in alignments:
                line += f"\t{label} {from_} {to}"

            output_dataset.append(line)

            save_result(os.path.join(args.save_path, filename), alignments, ocr)

    print("Inference done.")

    save_output_dataset(output_dataset, args.save_path)
    print("Output dataset saved.")

    return 0


if __name__ == "__main__":
    exit(main())
