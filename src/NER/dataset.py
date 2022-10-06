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

        self.load_data()

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

    def load_ocr(self, path):
        if self._txn is not None:
            text = self._txn.get(path.encode()).decode()

        else:
            with open(os.path.join(self.ocr_path, path), 'r') as f:
                text = f.read()

        return text

    def parse_annotation(self, line):
        file_path, *alignments = line.split("\t")
        return Annotation(file_path, self.load_ocr(file_path), self.parse_alignments(alignments))

    def parse_alignments(self, alignments):
        result = []

        for alignment in alignments:
            label, start, end,  = alignment.split()

            if label in self.model_config.labels:
                result.append(Alignment(label, int(start), int(end)))

        result.sort(key=lambda a: a.start)
        return result

    # https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=Eh3ckSO0YMZW
    def __getitem__(self, index):
        annotation = self.data[index]

        # TODO: Add different format tokenizers to helper and let user choose.
        tokens, iob = self.create_iob(annotation.text, annotation.alignments)

        encoding = self.tokenizer(tokens,
                                  padding="max_length",
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  truncation=True,
                                  max_length=self.model_config.max_sequence_len)

        labels = [self.model_config.labels2ids[label] for label in iob]

        encoded_labels = []

        # TODO: Move this to separate function
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

        # TODO: Move this to separate function
        if not self.model_config.sep_loss:
            for i, wordpiece in enumerate(self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])):
                encoded_labels[i] = encoded_labels[i] if wordpiece != "[LF]" else -100

        # TODO: Can this be done a bit cleaner?
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)
        item["ids"] = annotation.file_id

        return item

    def __len__(self):
        return len(self.data)

    # TODO: Move this to helper along with create_io? (which does not exist yet)
    def create_iob(self, text: str, alignments: typing.List[Alignment]):
        # beginning = text[:alignments[0].start].split()
        beginning = text[:alignments[0].start].replace("\n", " [LF] ").split()
        tokens = [[word, "O"] for word in beginning]

        for i, alignment in enumerate(alignments):
            # substr = text[alignment.start:alignment.end].split()
            substr = text[alignment.start:alignment.end].replace("\n", " [LF] ").split()
            tokens.extend([[word, alignment.label] for word in substr])

            try:
                next_alignment = alignments[i+1]
                # next_ = text[alignment.end:next_alignment.start].split()
                next_ = text[alignment.end:next_alignment.start].replace("\n", " [LF] ").split()
                tokens.extend([[word, "O"] for word in next_])
            except IndexError:
                end = text[alignment.end:].replace("\n", " [LF] ").split()
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
    
    example = dataset[0]
    print(example)

    ts = tokenizer.convert_ids_to_tokens(example["input_ids"])
    for t, l in zip(ts, example["labels"]):
        print(f"{t}\t{config.ids2labels[l.item()] if l != -100 else -100}")

    return 0


if __name__ == "__main__":
    exit(main())
