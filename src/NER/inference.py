# WARNING: At this moment, inference only works for standard BERT model
#          It does not work with lambert or bboxes

import os
import typing
import argparse
import numpy as np
import torch
import lmdb

import helper

from model import build_model

from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase

from src.alignment.timeout import timeout, TimeoutError


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", help="Path to a directory containing data for inference.", required=True)
    parser.add_argument("--config-path", help="Path to a json file contatining model config.", required=True)
    parser.add_argument("--model-path", help="Path to a trained model.", required=True)
    parser.add_argument("--tokenizer-path", help="Path to a tokenizer.", required=True)
    parser.add_argument("--save-path", help="Path to a output directory.", required=True)

    parser.add_argument("--save-all", help="Whether to save individual prediction files.", action="store_true", default=False)

    parser.add_argument("--print-conf", help="Add confidences to output files", action="store_true", default=False)
    parser.add_argument("--threshold", default=0.0, type=float, help="Threshold for selecting field with enough model confidence.")
    parser.add_argument("--aggfunc", default="prod", type=str, help="Confidence aggregation function.")

    parser.add_argument("--mode", type=int, default=0, choices=[0, 1], help="(0) Get data from folder structure OR (1) Get data from LMDB")
    parser.add_argument("--trn", help="Path to file containing training file names.")
    parser.add_argument("--val", help="Path to file containing validation file names.")
    parser.add_argument("--tst", help="Path to file containing test file names.")

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


def infer(ocr: str, tokenizer, model, filename) -> list:
    words = helper.add_line_separator_token(ocr).split() if model.config.sep else ocr.split()

    encoding = tokenizer(words,
                         padding="max_length",
                         is_split_into_words=True,
                         return_offsets_mapping=True,
                         max_length=512,
                         truncation=True,
                         return_tensors="pt")

    device = model.get_device()
    ids = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(ids, attention_mask=mask)[1][0]

    confidence = helper.calculate_confidence(logits).tolist()
    tokens, preds, confidence = convert_logits(model.config, tokenizer, logits, ids, confidence, encoding["offset_mapping"])

    if sum(encoding["attention_mask"].squeeze().tolist()) != 512:
        assert len(tokens) == len(preds), f"{filename}"
        assert len(tokens) == len(confidence), f"{filename}\t{len(tokens)}\t{len(confidence)}"
    else:
        print(f"File {filename} is being truncated due to maximum bert input length.")


    return list(zip(tokens, preds, confidence))


def convert_logits(model_config, tokenizer, logits, ids, confidence, offset_mapping):
    active_logits = logits.view(-1, model_config.num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)

    wordpieces = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    wordpieces_preds = [model_config.ids2labels[pred] for pred in flattened_predictions.cpu().numpy()]

    tokens = []
    preds = []
    confidences = [] # list of lists
    current_confidence = [] # list - confidences of subtokens of current tokens
    UNK = False

    for a, (tok, pred, conf, mapping) in enumerate(zip(wordpieces, wordpieces_preds, confidence, offset_mapping.squeeze().tolist())):
        if tok == helper.LINE_SEPARATOR or mapping[1] == 0 or tok == "[UNK]":
            UNK = True
            continue

        if UNK:
            mapping[0] = 0
            UNK = False

        if mapping[0] == 0 and mapping[1] != 0:
            if not current_confidence: # Current confidence is empty, we are starting a new token
                current_confidence.append(conf)
            else: # Current confidence is not empty, we are finishing a token
                confidences.append(current_confidence)
                current_confidence = []
                current_confidence.append(conf)

            tokens.append(tok)
            preds.append(pred)
        else:
            tokens[-1] += tok[2:] if tok.startswith("##") else tok
            current_confidence.append(conf)

    if current_confidence:
        confidences.append(current_confidence)

    return tokens, preds, confidences


@timeout(60)
def find_offsets(text: str, data: list, format: list) -> list:
    result = []

    char_index = 0
    token_index = 0

    # We have to normalize the text so we are able to find tokens in it
    # as tokenizers does normalization as well
    normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
    text_normalized = normalizer.normalize_str(text)

    while token_index < len(data):
        token, label, conf = data[token_index]

        if format == ["I", "O", "B"]:
            label = label if label == "O" else label[2:]

        text_slice = text_normalized[char_index:char_index+len(token)]

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
            result.append((label, from_, to, confidence))
            current_label = label
        else:
            _, current_from, _, confidences = result[-1]
            confidences.extend(confidence)
            result[-1] = (label, current_from, to, confidences)

    return result


def aggregate_confidence(alignments: list, aggfunc: typing.Callable) -> list:
    result = []

    for label, from_, to, confidences in alignments:
        result.append((label, from_, to, aggfunc(confidences).item(), confidences))

    return result


def save_result(path: str, result: list, ocr: str, print_conf: bool) -> None:
    with open(path, "w") as f:
        for label, from_, to, confidence, confidences in result:
            if print_conf:
                print(f"{label}\t{repr(ocr[from_:to])}\t{from_}\t{to}\t{confidence}\t{confidences}", file=f)
            else:
                print(f"{label}\t{repr(ocr[from_:to])}\t{from_}\t{to}", file=f)


def save_output_dataset(data: list, path) -> None:
    with open(path, "w") as f:
        for line in data:
            print(line, file=f)


def process_ocr(filename, ocr, tokenizer, model, aggfunc, threshold):
    # Get tokens, predictions and confidences
    result = infer(ocr, tokenizer, model, filename)

    # Get offsets of each token as [(token, label, start, end, confidence)]
    try:
        token_offsets = find_offsets(ocr, result, model.config.format)
    except TimeoutError:
        print(f"Timeout reached on file: {filename}")
        return []

    # Concat offsets of the same label as [(label, start, end, [confidences])]
    alignments = concat_token_offsets(token_offsets)

    # Filter out empty label
    alignments = [alignment for alignment in alignments if alignment[0] != "O"]

    alignments = aggregate_confidence(alignments, aggfunc)
    alignments = [alignment for alignment in alignments if alignment[3] >= threshold]
    
    return alignments


def run_mode0(args, tokenizer, model, aggfunc):
    n_inferred = 0
    output_dataset = []

    print("Starting inference ...")
    try:
        for root, _, filenames in os.walk(args.data_path):
            for filename in filenames:
                if not filename.endswith(".txt"):
                    continue
                file_path = os.path.join(root, filename)
                file_key = os.path.relpath(file_path, args.data_path)
                line = f"{file_key}"

                ocr = helper.load_ocr(file_path)

                alignments = process_ocr(file_key, ocr, tokenizer, model, aggfunc, args.threshold)

                for label, from_, to, _, _ in alignments:
                    line += f"\t{label} {from_} {to}"

                output_dataset.append(line)

                if args.save_all:
                    save_result(os.path.join(args.save_path, "files", file_key.replace("/", "-")), alignments, ocr, args.print_conf)

                n_inferred += 1
                if n_inferred % 100 == 0:
                    print(f"{n_inferred} files inferred.")

    except KeyboardInterrupt:
        pass

    print("Inference done.")

    save_output_dataset(output_dataset, os.path.join(args.save_path, "dataset.all"))
    print("Output dataset saved.")


def run_mode1(args, tokenizer, model, aggfunc):
    txn = lmdb.open(args.data_path, readonly=True, lock=False).begin()
    cursor = txn.cursor()
    print("LMDB connection open")

    output_trn = []
    output_val = []
    output_test = []
    output_other = []

    n_inferred = 0

    with open(args.trn, "r") as f:
        trn_files = [line.strip() for line in f]
    print("Training filenames loaded.")

    with open(args.val, "r") as f:
        val_files = [line.strip() for line in f]
    print("Validation filenames loaded.")

    with open(args.tst, "r") as f:
        test_files = [line.strip() for line in f]
    print("Test filenames loaded.")

    print(f"Inferring {len(trn_files)} train files")
    print(f"Inferring {len(val_files)} val files")
    print(f"Inferring {len(test_files)} test files")


    print("Starting inference ...")
    try:
        for filename, ocr in cursor:
            filename = filename.decode()
            ocr = ocr.decode()

            line = f"{filename}"

            alignments = process_ocr(filename, ocr, tokenizer, model, aggfunc, args.threshold)

            for label, from_, to, _, _ in alignments:
                line += f"\t{label} {from_} {to}"

            if filename in test_files:
                output_test.append(line)
            elif filename in trn_files:
                output_trn.append(line)
            elif filename in val_files:
                output_val.append(line)
            else:
                output_other.append(line)

            if args.save_all:
                save_result(os.path.join(args.save_path, "files", filename.replace("/", "-")), alignments, ocr, args.print_conf)

            n_inferred += 1
            if n_inferred % 100 == 0:
                print(f"{n_inferred} files inferred.")
    except KeyboardInterrupt:
        pass

    print("Inference done.")

    save_output_dataset(output_trn, os.path.join(args.save_path, "dataset.trn"))
    print("Training dataset saved.")
    save_output_dataset(output_val, os.path.join(args.save_path, "dataset.val"))
    print("Validation dataset saved.")
    save_output_dataset(output_test, os.path.join(args.save_path, "dataset.test"))
    print("Test dataset saved.")
    save_output_dataset(output_other, os.path.join(args.save_path, "dataset.other"))
    print("Remaining files dataset saved.")


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

    aggfunc = get_aggfunc(args.aggfunc)
    print(f"Aggregation function: {args.aggfunc}")
    print(f"Confidence threshold: {args.threshold}")

    os.makedirs(args.save_path, exist_ok=True)
    if args.save_all:
        os.makedirs(os.path.join(args.save_path, "files"), exist_ok=True)
    print("Output directory created.")

    run_func = run_mode0 if not args.mode else run_mode1
    run_func(args, tokenizer, model, aggfunc)
           
    return 0


if __name__ == "__main__":
    exit(main())
