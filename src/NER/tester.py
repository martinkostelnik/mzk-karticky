import torch

from helper import calculate_acc, IDS2LABELS

from sklearn.metrics import classification_report


class Tester:
    def __init__(self, model):
        self.model = model

        self.device = self.model.get_device()

    def step(self, batch):
        with torch.no_grad():
            loss, logits = self.forward(batch)

        return loss.item(), logits.detach()

    def forward(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        loss, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return loss, logits[0]

    def test(self, data_loader):
        self.model.eval()

        total_loss = 0
        total_acc = 0
        total_steps = 0

        truth = []
        prediction = []

        for batch in data_loader:
            loss, logits = self.step(batch)

            total_loss += loss
            acc, l, p = calculate_acc(batch["labels"], logits)

            total_acc += acc
            truth.extend([IDS2LABELS[id.item()] for id in l])
            prediction.extend([IDS2LABELS[id.item()] for id in p])

            total_steps += 1

        total_loss /= total_steps
        total_acc /= total_steps

        truth = [label if label == "O" else label[2:] for label in truth]
        prediction = [label if label == "O" else label[2:] for label in prediction]

        print(f"Test loss: {total_loss:.6f}")
        print(f"Test acc: {total_acc:.6f}\n")
        print(classification_report(truth, prediction, zero_division=0))
