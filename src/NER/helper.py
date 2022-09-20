import torch

from sklearn.metrics import accuracy_score

from dataset import AlignmentDataset


LABELS = ["Author", "Title", "Original_title", "Publisher", "Pages", "Series", "Edition", "References", "ID",
          "ISBN", "ISSN", "Topic", "Subtitle", "Date", "Institute", "Volume"]

FORMAT = ["B", "I"]

NUM_LABELS = len(LABELS) * len(FORMAT) + 1

LABELS2IDS = {f"{c}-{label}": i*len(FORMAT) + j + 1 for i, label in enumerate(LABELS) for j, c in enumerate(FORMAT)}
LABELS2IDS["O"] = 0

IDS2LABELS = {v: k for k, v in LABELS2IDS.items()}

MAX_TOKENS_LEN = 256

BERT_BASE_NAME = "bert-base-multilingual-uncased"


def load_dataset(data_path, ocr_path, batch_size, tokenizer, num_workers=0):
    dataset = AlignmentDataset(data_path=data_path, ocr_path=ocr_path, tokenizer=tokenizer)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader


def calculate_acc(labels, logits):
    flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
    active_logits = logits.view(-1, NUM_LABELS)  # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1).to(flattened_targets.device)  # shape (batch_size * seq_len,)
    active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

    labels_acc = torch.masked_select(flattened_targets, active_accuracy)
    predictions_acc = torch.masked_select(flattened_predictions, active_accuracy)

    return accuracy_score(labels_acc.cpu().numpy(), predictions_acc.cpu().numpy()), labels_acc, predictions_acc

def calculate_confidence(logits):
    fn = torch.nn.Softmax(dim=2)

    return fn(logits).cpu().numpy()
