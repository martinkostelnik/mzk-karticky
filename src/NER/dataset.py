import pandas as pd
import numpy as np
import typing
import os
import spacy
import torch

from spacy.training import offsets_to_biluo_tags
from transformers import BertTokenizerFast


# TODO: Import LABELS from helper?
LABELS = ["Author", "Title", "Original title", "Publisher", "Pages", "Series", "Edition", "References", "ID",
          "ISBN", "ISSN", "Topic", "Subtitle", "Date", "Institute", "Volume"]

FORMAT = ["B", "I"]

LABELS2IDS = {f"{c}-{label}": i*len(FORMAT) + j + 1 for i, label in enumerate(LABELS) for j, c in enumerate(FORMAT)}
LABELS2IDS["O"] = 0

IDS2LABELS = {v: k for k, v in LABELS2IDS.items()}


def prepare_training_data(ocr_path: str, alig_path: str) -> pd.DataFrame:
    """This function takes in the ocrs and alignments and creates a dataframe 
       containing two columns as (ocr, bilou-format).
    """

    res = []

    for root, dirs, files in os.walk(alig_path):
        for file in files:
            with open(os.path.join(root, file), "r") as f:
                offset_format = []

                for line in f:
                    s = line.split(chr(255))
                    offset_format.append((int(s[5]), int(s[7]), s[1]))

            with open(os.path.join(ocr_path, file), "r") as f:
                text = f.read()

            offset_format.sort(key=lambda x: x[0])

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

            item = ([token[0] for token in tokens], [token[1] for token in tokens])

            if len(item[0]) != len(item[1]):
                print("ERROR IN TOKENIZATION, LENGTHS DO NOT MATCH")

            res.append(item)

    return pd.DataFrame(res, columns=["text", "bilou"])


# TODO: Perhaps we should save the tokenizer (same as model) and not create new one for each DataSet instance?
class DataSet(torch.utils.data.Dataset):
    def __init__(self, df,  max_len: int=512):
        self.df = df
        self.len = len(df)
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")
        self.max_len=max_len

    # https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=Eh3ckSO0YMZW
    def __getitem__(self, index):
        text = self.df.text[index]
        bilou = self.df.bilou[index]

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
        
        del item["offset_mapping"]

        return item

    def __len__(self):
        return self.len
