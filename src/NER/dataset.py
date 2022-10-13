import typing
import os
import torch
import lmdb
import helper


class Alignment:
    def __init__(self, label: str, start: int, end: int):
        self.label = label
        self.start = start
        self.end = end

    def __str__(self):
        return f"{self.label}: {self.start}-{self.end}"


class Annotation:
    def __init__(self, file_id: str, text: str, alignments: list[Alignment]):
        self.file_id = file_id
        self.text = text
        self.alignments = alignments

    def __str__(self):
        output = f"{self.file_id}\n"
        output += f"{self.text}\n"
        output += '\n'.join([f"{alignment.label}: {alignment.start}-{alignment.end} [{self.text[alignment.start:alignment.end]}]" for alignment in self.alignments])
        return output


class AlignmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        ocr_path: str,
        tokenizer,
        model_config = helper.ModelConfig()
    ):
        self.data: typing.List[Annotation] = []
        self.tokenizer = tokenizer
        self.model_config = model_config

        self.data_path = data_path
        self.ocr_path = ocr_path

        self._txn = lmdb.open(self.ocr_path, readonly=True, lock=False).begin() if "lmdb" in self.ocr_path else None

        self.formatting_function = self.get_formatting_function()

        self.load_data()

    def get_formatting_function(self):
        if self.model_config.format == ["I", "O", "B"]:
            return helper.offsets_to_iob

        if self.model_config.format == ["I", "O"]:
            return helper.offsets_to_io

        raise ValueError("Invalid formatting functionn name. Must be 'iob' or 'io'.")

    def load_data(self) -> None:
        with open(self.data_path) as file:
            for line in file:
                line = line.strip()
                if len(line) > 0:
                    annotation = self.parse_annotation(line)
                    
                    # TODO: Make it more modular, let user select how many alignments an annotation must have
                    # and what must be the subset
                    if len(annotation.alignments) >= 4 and set(["Author", "ID", "Title"]).issubset(set([alignment.label for alignment in annotation.alignments])):
                        self.data.append(annotation)

    def parse_annotation(self, line):
        file_path, *alignments = line.split("\t")
        path =  file_path if self._txn else os.path.join(self.ocr_path, file_path)
        return Annotation(file_path, helper.load_ocr(path=path, txn=self._txn), self.parse_alignments(alignments))

    def parse_alignments(self, alignments):
        result = []

        for alignment in alignments:
            label, start, end,  = alignment.split()

            if label in self.model_config.labels:
                result.append(Alignment(label, int(start), int(end)))

        result.sort(key=lambda a: a.start)
        return result

    def encode_labels(self, label_ids: list, offset_mapping):
        encoded_labels = []

        current_label = label_ids[0]
        i = 0
        for mapping in offset_mapping:
            if mapping[1] == 0:
                encoded_labels.append(-100)
            elif mapping[0] == 0: # First is zero, second is not zero => we are at first subtoken of token
                current_label = label_ids[i]
                i += 1
                encoded_labels.append(current_label)
            else: # Both are non-zero => we are at non-first subtoken of token
                encoded_labels.append(current_label)

        return encoded_labels

    def disable_loss_on_sep(self, labels, input_ids):
        return [labels[i] if wordpiece != helper.LINE_SEPARATOR else -100 for i, wordpiece in enumerate(self.tokenizer.convert_ids_to_tokens(input_ids))]

    def __getitem__(self, index):
        annotation = self.data[index]

        tokens, labels = self.formatting_function(annotation.text, annotation.alignments)

        encoding = self.tokenizer(tokens,
                                  padding="max_length",
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  truncation=True,
                                  max_length=self.model_config.max_sequence_len)

        label_ids = [self.model_config.labels2ids[label] for label in labels]

        encoded_labels = self.encode_labels(label_ids, encoding["offset_mapping"])

        # Disable loss calculation on line separation token
        if not self.model_config.sep_loss:
            encoded_labels = self.disable_loss_on_sep(encoded_labels, encoding["input_ids"])

        # Create tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)
        item["ids"] = annotation.file_id

        return item

    def __len__(self):
        return len(self.data)


def load_dataset(data_path, ocr_path, batch_size, tokenizer, model_config=helper.ModelConfig(), num_workers=0):
    dataset = AlignmentDataset(data_path=data_path, ocr_path=ocr_path, tokenizer=tokenizer, model_config=model_config)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help="Path to the alignments file.")
    parser.add_argument("--ocr-path", help="Path to either (1) dir with OCR files or (2) LMDB with texts from OCR.")
    parser.add_argument("--tokenizer-path", help="Path to the tokenizer.")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    config = helper.ModelConfig()

    tokenizer = helper.build_tokenizer(path=args.tokenizer_path, model_config=config)
    dataset = AlignmentDataset(args.data_path, args.ocr_path, tokenizer=tokenizer, model_config=config)

    print(f"Dataset size: {len(dataset)}")
    
    example = dataset[29]
    print(example)

    ts = tokenizer.convert_ids_to_tokens(example["input_ids"])
    for t, l in zip(ts, example["labels"]):
        print(f"{t}\t{config.ids2labels[l.item()] if l != -100 else -100}")

    return 0


if __name__ == "__main__":
    exit(main())
