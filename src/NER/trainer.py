import typing
import argparse
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from dataset import DataSet, IDS2LABELS
from model import NerBert
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report


# TODO: Split into train and test test.
class Trainer:
    def __init__(self, data: pd.DataFrame, settings: dict, model: NerBert=NerBert()):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Split dataset into train, val and test
        np.random.seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = np.split(data.sample(frac=1, random_state=42), [int(settings["train_ratio"]*len(data)), int((1 - settings["val_ratio"])*len(data))])
        self.train_dataset = self.train_dataset.reset_index()
        self.val_dataset = self.val_dataset.reset_index()
        self.test_dataset = self.test_dataset.reset_index()
        
        # Set training settings
        self.epochs = settings["epochs"]
        self.batch_size = settings["batch_size"]
        self.model = model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=settings["learning_rate"])
        self.max_norm = settings["max_grad_norm"]

    def train(self):
        self.model.train()
        
        # Prepare data loaders
        train_dataset = DataSet(self.train_dataset)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        val_dataset = DataSet(self.val_dataset)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True,num_workers=0)

        best_epoch = np.inf

        # Start training
        for epoch in range(self.epochs):
            epoch_acc_train = 0
            epoch_loss_train = 0
            epoch_acc_val = 0
            epoch_loss_val = 0

            train_steps = 0
            val_steps = 0

            # Training loop
            self.model.train()
            for batch in train_data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels)
                loss, logits = outputs[0], outputs[1]

                epoch_loss_train += loss.item()
                epoch_acc_train += self.calculate_acc(labels, logits)[0]
                
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_steps += 1

            # Validation loop
            self.model.eval()
            with torch.no_grad():
                for batch in val_data_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(input_ids, attention_mask, labels)
                    loss, logits = outputs[0], outputs[1]

                    epoch_loss_val += loss.item()
                    epoch_acc_val += self.calculate_acc(labels, logits)[0]

                    val_steps += 1

            if epoch_loss_val / val_steps < best_epoch:
                torch.save(self.model.state_dict(), r"model/nerbert.pt")
                best_epoch = epoch_loss_val / val_steps

            print(f"Epoch {epoch+1} | Loss: {epoch_loss_train / train_steps} | Acc: {epoch_acc_train / train_steps} | Val_Loss: {epoch_loss_val / val_steps} | Val_Acc: {epoch_acc_val / val_steps}")

    def evaluate(self):
        self.model.eval()

        # Preapre loader
        test_dataset = DataSet(self.test_dataset)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        test_loss = 0
        test_acc = 0

        steps = 0

        report_labels = []
        report_preds = []

        # Test loop
        with torch.no_grad():
            for batch in test_data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels)
                loss, logits = outputs[0], outputs[1]

                test_loss += loss.item()
                acc, l, p = self.calculate_acc(labels, logits)

                report_labels.append(l)
                report_preds.append(p)

                test_acc += acc
                
                steps += 1

        # TODO: This might not be correct, but it throws no exceptions in classification_report()
        l = [[IDS2LABELS[id.item()] for id in l] for l in report_labels]
        p = [[IDS2LABELS[id.item()] for id in l] for l in report_preds]

        print(f"Test loss: {test_loss / steps}")
        print(f"Test acc: {test_acc / steps}")
        print(classification_report(l, p, zero_division=0))

    def calculate_acc(self, labels, logits):
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = logits.view(-1, 65) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
        labels_acc = torch.masked_select(flattened_targets, active_accuracy)
        predictions_acc = torch.masked_select(flattened_predictions, active_accuracy)

        return accuracy_score(labels_acc.cpu().numpy(), predictions_acc.cpu().numpy()), labels_acc, predictions_acc