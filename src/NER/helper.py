import torch
import numpy as np
import typing
import json
import os

from sklearn.metrics import accuracy_score

from transformers import BertTokenizerFast

JOKER = chr(65533)
UNICODE_JOKER = chr(772)
LINE_SEPARATOR = "[LF]"
BERT_BASE_NAME = "bert-base-multilingual-uncased"


class ModelConfig:
    ALL_LABELS = ["Author", "Title", "Original_title", "Publisher", "Pages", "Series", "Edition", "References", "ID", "ISBN", "ISSN", "Topic", "Subtitle", "Date", "Institute", "Volume"]
    FILENAME = r"model_config.json"

    def __init__(
        self,
        labels: str = "all",
        format: str = "iob",
        max_len: int = 256,
        sep: bool = True,
        sep_loss: bool = False,
    ):
        self.labels = self.get_labels(labels)
        self.format = self.get_format(format)
        self.num_labels = len(self.labels) * len(self.format) + 1

        self.labels2ids = {f"{c}-{label}": i*len(self.format) + j + 1 for i, label in enumerate(self.labels) for j, c in enumerate(self.format)}
        self.labels2ids["O"] = 0
        self.ids2labels = {v: k for k, v in self.labels2ids.items()}

        self.max_sequence_len = max_len

        self.sep = sep
        self.sep_loss = sep_loss

    def save(self, path: str):
        path = os.path.join(path, self.FILENAME)

        json_obj = json.dumps(self.__dict__, indent=4)

        with open(path, "w") as f:
            f.write(json_obj)

    @classmethod
    def load(cls, path: str):
        path = os.path.join(path, cls.FILENAME)
        
        with open(path, "r") as f:
            data = json.load(f)

        config = ModelConfig()

        for key, val in data.items():
            setattr(config, key, val)

        # Change data type of ids2labels keys to int
        config.ids2labels = {int(v): k for v, k in config.ids2labels.items()}

        return config

    def get_format(self, format: str):
        return list(format.upper())

    def get_labels(self, label_str: str):
        if label_str == "all":
            return self.ALL_LABELS

        if label_str == "subset":
            return ["Author", "Title", "ID", "Pages", "Volume", "Publisher", "Edition", "Date"]

    def __str__(self):
        output = f"Labels used: {self.labels}\n"
        output += f"Number of labels: {self.num_labels}\n"
        output += f"Format used: {self.format}\n"

        output += f"labels2ids: {self.labels2ids}\n"
        output += f"ids2labels: {self.ids2labels}\n"

        output += f"Max seq length: {self.max_sequence_len}\n"

        output += f"Separating: {self.sep}\n"
        output += f"Loss on sep: {self.sep_loss}\n"
        
        return output


def calculate_acc(labels, logits, num_labels):
    flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
    active_logits = logits.view(-1, num_labels)  # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1).to(flattened_targets.device)  # shape (batch_size * seq_len,)
    active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

    labels_acc = torch.masked_select(flattened_targets, active_accuracy)
    predictions_acc = torch.masked_select(flattened_predictions, active_accuracy)

    return accuracy_score(labels_acc.cpu().numpy(), predictions_acc.cpu().numpy()), labels_acc, predictions_acc


def calculate_confidence(logits):
    return torch.nn.functional.softmax(logits, dim=2).cpu().numpy().max(axis=2).flatten()


def build_tokenizer(path: str, model_config: ModelConfig=ModelConfig()):
    if path == BERT_BASE_NAME:
        tokenizer = BertTokenizerFast.from_pretrained(BERT_BASE_NAME)
        tokenizer.add_special_tokens({"additional_special_tokens": [JOKER, UNICODE_JOKER]})

        if model_config.sep:
            tokenizer.add_special_tokens({"additional_special_tokens": [LINE_SEPARATOR]})
            
        return tokenizer

    return BertTokenizerFast.from_pretrained(path)


def offsets_to_io(text: str, alignments):
    # TODO: This should be modular, based on model config. Do it in dataset.py maybe?
    text_c = text.replace("\n", f" {LINE_SEPARATOR} ")

    tokens = text_c[:alignments[0].start].split()
    labels = ["O"] * len(tokens)

    for i, alignment in enumerate(alignments):
        tokens += text_c[alignment.start:alignment.end].split()
        labels += [alignment.label] * (len(tokens) - len(labels))

        try:
            next_alignment = alignments[i + 1]
            tokens += text_c[alignment.end:next_alignment.start].split()
        except IndexError:
            tokens += text_c[alignment.end:].split()

        labels += ["O"] * (len(tokens) - len(labels))

    return tokens, labels


def offsets_to_iob(text: str, alignments):
    tokens, labels = offsets_to_io(text, alignments)

    current = ""
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == "O":
            current = "O"
            continue

        tmp = label
        labels[i] = f"B-{label}" if label != current else f"I-{label}"
        current = tmp

    return tokens, labels
