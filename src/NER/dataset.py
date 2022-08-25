import numpy as np
import typing
import os
import zipfile
import torch
import json
import sys
from tqdm import tqdm

import pandas as pd


LABELS = ["Author", "Title", "Original title", "Publisher", "Pages", "Series", "Edition", "References", "ID",
          "ISBN", "ISSN", "Topic", "Subtitle", "Date", "Institute", "Volume"]

FORMAT = ["B", "I"]

LABELS2IDS = {f"{c}-{label}": i*len(FORMAT) + j + 1 for i, label in enumerate(LABELS) for j, c in enumerate(FORMAT)}
LABELS2IDS["O"] = 0

IDS2LABELS = {v: k for k, v in LABELS2IDS.items()}


class FullDataset(torch.utils.data.Dataset):
    def __init__(self, ocr_path: str, alig_path: str, tokenizer, debug: bool, max_len: int=256):
        self.data = self.load_data(ocr_path, alig_path)
        self.len = len(self.data)
        self.tokenizer = tokenizer
        self.max_len = max_len

        if debug:
            self.print_dataset_debug()

    def load_data(self, ocr_path: str, alig_path: str) -> list:
        # Delete items present in test dataset
        test_dataset = []
        with open("mapping.txt", "r") as f:
            for line in f:
                test_dataset.append(line.split()[0].replace("-", "/"))

        res = []
        with zipfile.ZipFile(alig_path, "r") as z:
            for name in z.namelist():
                if not name.endswith(".txt") or name.partition("/")[2] in test_dataset or "samples" in name:
                    continue

                offset_format = []

                with z.open(name) as f:
                    for line in f:
                        s = line.decode("utf-8").split("\t")
                        # (from, to, label)
                        offset_format.append((int(s[2]), int(s[3]), s[0]))

                with open(os.path.join(ocr_path, name.partition("/")[2]), 'r') as f:
                    text = f.read()

                offset_format.sort(key=lambda x: x[0])
                
                if len(offset_format) > 0 and offset_format[-1][1] < len(text):
                    res.append((text, offset_format))

        return res

    # https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=Eh3ckSO0YMZW
    def __getitem__(self, index):
        text, offset_format = self.data[index]

        tokens, iob = self.create_iob(text, offset_format)

        encoding = self.tokenizer(tokens,
                                  padding="max_length",
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  truncation=True,
                                  max_length=self.max_len)

        labels = [LABELS2IDS[label] for label in iob]

        encoded_labels = []

        current_label = labels[0]
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[1] == 0:
                encoded_labels.append(-100)
            elif mapping[0] == 0:
                current_label = labels[i]
                i += 1
                encoded_labels.append(current_label)
            else:
                encoded_labels.append(current_label)

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len

    def create_iob(self, text: str, offset_format: list):
        beginning = text[0:offset_format[0][0]].split()
        tokens = [[word, "O"] for word in beginning]

        for i, item in enumerate(offset_format):
            substr = text[item[0]:item[1]].split()

            tokens.extend([[word, item[2]] for word in substr])

            try:
                next_ = text[item[1]:offset_format[i+1][0]].split()
                tokens.extend([[word, "O"] for word in next_])
            except IndexError:
                end = text[item[1]:].split()
                tokens.extend([[word, "O"] for word in end])

        current = ""
        for token in tokens:
            if token[1] == "O":
                current = "O"
                continue

            tmp = token[1]
            token[1] = f"B-{token[1]}" if token[1] != current else f"I-{token[1]}"
            current = tmp

        tokens, iob = ([token[0] for token in tokens], [token[1] for token in tokens])

        return tokens, iob

    def print_dataset_debug(self):
        idx = np.random.randint(low=0, high=self.len - 1)
        example_raw = self.data[idx]
        example = self.__getitem__(idx)

        with open("debug.txt", "w") as f:
            print("----------------------------------------------------------------------------------------------------------------------------------\n", file=f)
            print("Example raw data point in Pytorch Dataset:\n", file=f)
            print(example_raw[0], file=f)

            for item in example_raw[1]:
                print(f"{item} {repr(example_raw[0][item[0]:item[1]])}", file=f)
                
            print("\n----------------------------------------------------------------------------------------------------------------------------------\n", file=f)

            print("Example data point from tokenizer:\n", file=f)
            print("_______________________________tokenizer________________________________|___________truth___________", file=f)
            print("input_ids       token_type_ids  attention_mask  offset_mapping  labels  |  labels            tokens", file=f)

            tokens = self.tokenizer.convert_ids_to_tokens(example["input_ids"].squeeze().tolist())

            labels = []
            for label in example["labels"]:
                try:
                    labels.append(IDS2LABELS[label.item()])
                except KeyError:
                    labels.append("-")

            for input_id, type_id, attention_mask, offset, label, text_label, token in zip(example["input_ids"], example["token_type_ids"], example["attention_mask"], example["offset_mapping"], example["labels"], labels, tokens):
                print(f"{input_id.item():<14}  {type_id.item():<14}  {attention_mask.item():<14}  {str(offset.tolist()):<14}  {label.item():<8}|  {text_label:<16}  {token}", file=f)

            print("\n----------------------------------------------------------------------------------------------------------------------------------\n", file=f)

            sums = [torch.sum(dato["attention_mask"]).item() for dato in self]
            pd.DataFrame(sums).to_csv("sums.txt", index=False, header=False)
            max_len = max(sums)
            print(f"Longest token sequence in data is {max_len} tokens long.", file=f)
            print(f"Max token sequence length is currently set to {self.max_len}.", file=f)
           
            if max_len == self.max_len:
                for handle in [f, sys.stderr]:
                    print(f"WARNING: Longest sequence ({max_len}) should not have the same length as maximum length.", file=handle)
                    print(f"This probably means that data are getting truncated.", file=handle)
                    print(f"You should increase maximum length and re-check the longest sequence found.", file=handle)


class HandAnnotatedDataset(FullDataset):
    def __init__(self, ocr_path: str, alig_path: str, tokenizer, max_len: int=256):
        self.data = self.load_data(ocr_path, alig_path)
        self.len = len(self.data)
        self.tokenizer = tokenizer
        self.max_len=max_len

    def load_data(self, ocr_path: str, alig_path: str) -> list:
        res = []

        with open(alig_path, "r") as json_f:
            json_data = json.load(json_f)

        for annotated_file in json_data:
            if "label" in annotated_file:
                offset_format = []

                filename = annotated_file["text"].rpartition("/")[2]

                for annotation in annotated_file["label"]:
                    label = annotation["labels"][0]
                    start = annotation["start"]
                    end = annotation["end"]

                    offset_format.append((start, end, label))

                with open(os.path.join(ocr_path, filename), 'r') as f:
                    text = f.read()

                offset_format.sort(key=lambda x: x[0])
            
                res.append((text, offset_format))

        return res
