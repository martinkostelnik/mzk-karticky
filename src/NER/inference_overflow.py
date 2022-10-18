import argparse
import torch
import helper
from model import build_model
import os
import pandas as pd
from dataset import AlignmentDataset


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", help="Path to a directory containing data for inference.", required=True)
    parser.add_argument("--config-path", help="Path to a json file contatining model config.", required=True)
    parser.add_argument("--model-path", help="Path to a trained model.", required=True)
    parser.add_argument("--tokenizer-path", help="Path to a tokenizer.", required=True)

    args = parser.parse_args()
    return args


def inferWhole(tokens, tokenizer, model):
    encoding = tokenizer(tokens,
                         is_split_into_words=True,
                         return_offsets_mapping=True,
                         max_length=512,
                         padding="max_length",
                         return_tensors="pt")

    device = model.get_device()
    ids = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(ids, attention_mask=mask)[1][0]

    preds = logits_to_tokens_preds(model.config, tokenizer, logits, ids, encoding["offset_mapping"])

    return preds


def inferParts(tokens, tokenizer, model):
    split_words = [tokens[:100], tokens[100:]]

    out_preds = []

    for part in split_words:
        encoding = tokenizer(part,
                            is_split_into_words=True,
                            return_offsets_mapping=True,
                            max_length=256,
                            padding="max_length",
                            return_tensors="pt")

        device = model.get_device()
        ids = encoding["input_ids"].to(device)
        mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(ids, attention_mask=mask)[1][0]

        preds = logits_to_tokens_preds(model.config, tokenizer, logits, ids, encoding["offset_mapping"])
        out_preds.extend(preds)

    return out_preds

def logits_to_tokens_preds(model_config, tokenizer, logits, ids, offset_mapping):
    active_logits = logits.view(-1, model_config.num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)

    wordpieces = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    wordpieces_preds = [model_config.ids2labels[pred] for pred in flattened_predictions.cpu().numpy()]

    preds = []

    for tok, pred, mapping in zip(wordpieces, wordpieces_preds, offset_mapping.squeeze().tolist()):
        if mapping[1] == 0:
            continue

        if mapping[0] == 0:
            preds.append(pred)

    return preds


def long_count():
    df = pd.read_csv("/home/xkoste12/mzk-karticky/data/token_lengths.txt", sep="\t", header=None, names=["file", "len"])

    total = len(df)
    df_long = df[df["len"] >= 256]
    df_long = df_long[df["len"] <= 512].reset_index().sort_values(by="len", ascending=False)
    long = len(df_long)

    return df_long, f"{long}/{total}\t{long/total*100} %"


def get_alignments(l: list):
    path = r"/mnt/xkoste12/matylda1/ikiss/data/mzk_karticky/2022-09-02/alignment.all"
    result = []

    with open(path, "r") as f:
        for line in f:
            filename, *_ = line.split("\t")
            if filename in l:
                result.append(line)
    
    return result

def main():
    args = parse_arguments()

    df, count = long_count()

    # result = get_alignments(df[:25]["file"].tolist())
    # for line in result:
    #     print(line, end="") 
    print(count)

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

    dataset = AlignmentDataset("/home/xkoste12/mzk-karticky/data/alignments.long",
                               "/mnt/xkoste12/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-ocr",
                               tokenizer=tokenizer,
                               model_config=model_config)
    print("Dataset loaded")

    print("Starting ...")
    for i, dato in enumerate(dataset.data):
        tokens, labels = helper.offsets_to_iob(dato.text, dato.alignments, True)

        preds_whole = inferWhole(tokens, tokenizer, model)
        preds_parts = inferParts(tokens, tokenizer, model)

        assert len(tokens) == len(labels)
        assert len(labels) == len(preds_whole)
        assert len(labels) == len(preds_parts)

        error_w = sum([label != pred for label, pred in zip(labels, preds_whole)])
        error_p = sum([label != pred for label, pred in zip(labels, preds_parts)])

        print(f"{dato.file_id}\tWHOLE: {error_w}\tPARTS: {error_p}\t({len(tokens)})")
            
if __name__ == "__main__":
    exit(main())
