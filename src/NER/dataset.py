import typing
import os
import torch
import lmdb

import helper


class Annotation:
    def __init__(self, file_id, text, alignments):
        self.file_id = file_id
        self.text = text
        self.alignments = alignments

    def __str__(self):
        output = f"{self.file_id}\n"
        output += f"{self.text}\n"
        output += '\n'.join([f"{alignment.label}: {alignment.start}-{alignment.end} [{self.text[alignment.start:alignment.end]}]" for alignment in self.alignments])
        return output


class Alignment:
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end

    def __str__(self):
        return f"{self.label}: {self.start}-{self.end}"


class AlignmentDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, ocr_path: str, tokenizer):
        self.data: typing.List[Annotation] = []
        self.tokenizer = tokenizer
        self.max_len = helper.MAX_TOKENS_LEN

        self.data_path = data_path
        self.ocr_path = ocr_path

        self._txn = lmdb.open(self.ocr_path, readonly=True, lock=False).begin() if "lmdb" in self.ocr_path else None

        self.load_data()

    def load_data(self) -> None:
        with open(self.data_path) as file:
            for line in file:
                line = line.strip()
                if len(line) > 0:
                    self.data.append(self.parse_annotation(line))

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
            result.append(Alignment(label, int(start), int(end)))

        result.sort(key=lambda a: a.start)
        return result

    # https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=Eh3ckSO0YMZW
    def __getitem__(self, index):
        annotation = self.data[index]
        tokens, iob = self.create_iob(annotation.text, annotation.alignments)

        encoding = self.tokenizer(tokens,
                                  padding="max_length",
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  truncation=True,
                                  max_length=self.max_len)

        labels = [helper.LABELS2IDS[label] for label in iob]

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
        item["ids"] = annotation.file_id

        return item

    def __len__(self):
        return len(self.data)

    def create_iob(self, text: str, alignments: typing.List[Alignment]):
        beginning = text[:alignments[0].start].split()
        tokens = [[word, "O"] for word in beginning]

        for i, alignment in enumerate(alignments):
            substr = text[alignment.start:alignment.end].split()

            tokens.extend([[word, alignment.label] for word in substr])

            try:
                next_alignment = alignments[i+1]
                next_ = text[alignment.end:next_alignment.start].split()
                tokens.extend([[word, "O"] for word in next_])
            except IndexError:
                end = text[alignment.end:].split()
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

    # def print_dataset_debug(self):
    #     idx = np.random.randint(low=0, high=len(self) - 1)
    #     example_raw = self.data[idx]
    #     example = self.__getitem__(idx)
    #
    #     with open("debug.txt", "w") as f:
    #         print("----------------------------------------------------------------------------------------------------------------------------------\n", file=f)
    #         print("Example raw data point in Pytorch Dataset:\n", file=f)
    #         print(example_raw[0], file=f)
    #
    #         for item in example_raw[1]:
    #             print(f"{item} {repr(example_raw[0][item[0]:item[1]])}", file=f)
    #
    #         print("\n----------------------------------------------------------------------------------------------------------------------------------\n", file=f)
    #
    #         print("Example data point from tokenizer:\n", file=f)
    #         print("_______________________________tokenizer________________________________|___________truth___________", file=f)
    #         print("input_ids       token_type_ids  attention_mask  offset_mapping  labels  |  labels            tokens", file=f)
    #
    #         tokens = self.tokenizer.convert_ids_to_tokens(example["input_ids"].squeeze().tolist())
    #
    #         labels = []
    #         for label in example["labels"]:
    #             try:
    #                 labels.append(IDS2LABELS[label.item()])
    #             except KeyError:
    #                 labels.append("-")
    #
    #         for input_id, type_id, attention_mask, offset, label, text_label, token in zip(example["input_ids"], example["token_type_ids"], example["attention_mask"], example["offset_mapping"], example["labels"], labels, tokens):
    #             print(f"{input_id.item():<14}  {type_id.item():<14}  {attention_mask.item():<14}  {str(offset.tolist()):<14}  {label.item():<8}|  {text_label:<16}  {token}", file=f)
    #
    #         print("\n----------------------------------------------------------------------------------------------------------------------------------\n", file=f)
    #
    #         sums = [torch.sum(dato["attention_mask"]).item() for dato in self]
    #         pd.DataFrame(sums).to_csv("sums.txt", index=False, header=False)
    #         max_len = max(sums)
    #         print(f"Longest token sequence in data is {max_len} tokens long.", file=f)
    #         print(f"Max token sequence length is currently set to {self.max_len}.", file=f)
    #
    #         if max_len == self.max_len:
    #             for handle in [f, sys.stderr]:
    #                 print(f"WARNING: Longest sequence ({max_len}) should not have the same length as maximum length.", file=handle)
    #                 print(f"This probably means that data are getting truncated.", file=handle)
    #                 print(f"You should increase maximum length and re-check the longest sequence found.", file=handle)


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help="Path to the alignments file.")
    parser.add_argument("--ocr-path", help="Path to either (1) dir with OCR files or (2) LMDB with texts from OCR.")
    parser.add_argument("--tokenizer-path", help="Path to the tokenizer.")

    args = parser.parse_args()
    return args


def main():
    from transformers import BertTokenizerFast

    args = parse_arguments()

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)
    dataset = AlignmentDataset(args.data_path, args.ocr_path, tokenizer=tokenizer)

    print(f"Dataset size: {len(dataset)}")
    print(dataset[0])

    return 0


if __name__ == "__main__":
    exit(main())
