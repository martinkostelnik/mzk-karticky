import pandas as pd
import numpy as np
import typing
import os
import zipfile
import torch

from transformers import BertTokenizerFast
from tqdm import tqdm


LABELS = ["Author", "Title", "Original title", "Publisher", "Pages", "Series", "Edition", "References", "ID",
          "ISBN", "ISSN", "Topic", "Subtitle", "Date", "Institute", "Volume"]

FORMAT = ["B", "I"]

LABELS2IDS = {f"{c}-{label}": i*len(FORMAT) + j + 1 for i, label in enumerate(LABELS) for j, c in enumerate(FORMAT)}
LABELS2IDS["O"] = 0

IDS2LABELS = {v: k for k, v in LABELS2IDS.items()}


class DataSet(torch.utils.data.Dataset):
    def __init__(self, ocr_path: str, alig_path: str, tokenizer, max_len: int=512):
        self.data = self.load_data(ocr_path, alig_path)
        self.len = len(self.data)
        self.tokenizer = tokenizer
        self.max_len=max_len

    def load_data(self, ocr_path: str, alig_path: str) -> pd.DataFrame:
        res = []
        with zipfile.ZipFile(alig_path, "r") as z:
            for name in tqdm(z.namelist(), desc="Preparing data"):
                if name[-4:] != ".txt":
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

        return res#pd.DataFrame(res, columns=["text", "alignment"], dtype=[("text", str), ("alignment", list)])

    # https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=Eh3ckSO0YMZW
    def __getitem__(self, index):
        text, offset_format = self.data[index]

        ########################
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

        text, bilou = ([token[0] for token in tokens], [token[1] for token in tokens])
        
        #########################

        encoding = self.tokenizer(text,
                                  padding="max_length",
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  truncation=True,
                                  max_length=self.max_len)

        labels = [LABELS2IDS[label] for label in bilou]

        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            encoded_labels[idx] = labels[i]
            i += 1

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)
        
        #del item["offset_mapping"]

        return item

    def __len__(self):
        return self.len






# TMP DataSet to use for hand-annotated
import json
class DataSetHandy(torch.utils.data.Dataset):
    def __init__(self, ocr_path: str, alig_path: str, tokenizer, max_len: int=512):
        self.data = self.load_data(ocr_path, alig_path)
        self.len = len(self.data)
        self.tokenizer = tokenizer
        self.max_len=max_len

    def load_data(self, ocr_path: str, alig_path: str) -> pd.DataFrame:
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

    # https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=Eh3ckSO0YMZW
    def __getitem__(self, index):
        text, offset_format = self.data[index]

        ########################
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

        text, bilou = ([token[0] for token in tokens], [token[1] for token in tokens])
        
        #########################

        encoding = self.tokenizer(text,
                                  padding="max_length",
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  truncation=True,
                                  max_length=self.max_len)

        labels = [LABELS2IDS[label] for label in bilou]

        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            encoded_labels[idx] = labels[i]
            i += 1

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)
        
        #del item["offset_mapping"]

        return item

    def __len__(self):
        return self.len
