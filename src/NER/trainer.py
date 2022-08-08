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


class Trainer:
    def __init__(self, settings: dict, model, tokenizer):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = tokenizer

        # Split dataset into train, val and test
        # np.random.seed(42)
        # self.train_dataset, self.val_dataset, self.test_dataset = np.split(data.sample(frac=1, random_state=42), [int(settings["train_ratio"]*len(data)), int((1 - settings["val_ratio"])*len(data))])
        # self.train_dataset = self.train_dataset.reset_index()
        # self.val_dataset = self.val_dataset.reset_index()
        # self.test_dataset = self.test_dataset.reset_index()
        
        # Set training settings
        self.epochs = settings["epochs"]
        # self.batch_size = settings["batch_size"]
        self.model = model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=settings["learning_rate"])
        self.max_norm = settings["max_grad_norm"]
        self.num_labels = settings["num_labels"]
        self.examples = settings["examples"]

        # Disable BERT training
        if not settings["bert"]:
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def train(self, train_data_loader, val_data_loader):
        self.model.train()
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
            for i, batch in enumerate(train_data_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs[0], outputs[1]

                epoch_loss_train += loss.item()
                epoch_acc_train += self.calculate_acc(labels, logits)[0]
                
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_steps += 1

                if i == 0:
                    example_ids = input_ids
                    example_logits = logits
                    example_labels = labels
                    example_offset_mapping = batch["offset_mapping"]

            # Validation loop
            self.model.eval()
            with torch.no_grad():
                for batch in val_data_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss, logits = outputs[0], outputs[1]

                    epoch_loss_val += loss.item()
                    epoch_acc_val += self.calculate_acc(labels, logits)[0]

                    val_steps += 1

            if epoch_loss_val / val_steps < best_epoch:
                self.model.save_pretrained(r"model")
                self.tokenizer.save_vocabulary(r"model")
                best_epoch = epoch_loss_val / val_steps

            print(f"Epoch {epoch+1} | Loss: {epoch_loss_train / train_steps} | Acc: {epoch_acc_train / train_steps} | Val_Loss: {epoch_loss_val / val_steps} | Val_Acc: {epoch_acc_val / val_steps}")
            
            if self.examples:
                self.print_epoch_example(example_logits, example_labels, example_ids, example_offset_mapping)

    def evaluate(self, test_data_loader):
        self.model.eval()

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

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs[0], outputs[1]

                test_loss += loss.item()
                acc, l, p = self.calculate_acc(labels, logits)

                report_labels.append(l)
                report_preds.append(p)

                test_acc += acc
                
                steps += 1

        l = [[IDS2LABELS[id.item()] for id in l_] for l_ in report_labels]
        p = [[IDS2LABELS[id.item()] for id in p_] for p_ in report_preds]

        print(f"Test loss: {test_loss / steps}")
        print(f"Test acc: {test_acc / steps}")
        print(classification_report(l, p, zero_division=0))

    def calculate_acc(self, labels, logits):
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = logits.view(-1, self.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
        labels_acc = torch.masked_select(flattened_targets, active_accuracy)
        predictions_acc = torch.masked_select(flattened_predictions, active_accuracy)

        return accuracy_score(labels_acc.cpu().numpy(), predictions_acc.cpu().numpy()), labels_acc, predictions_acc

    def print_epoch_example(self, logits_, labels_, ids_, offset_mapping_):
        for logits, labels, ids, offset_mapping in zip(logits_, labels_, ids_, offset_mapping_):
            active_logits = logits.view(-1, self.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)

            tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
            token_predictions = [IDS2LABELS[i] for i in flattened_predictions.cpu().numpy()]
            wp_preds = list(zip(tokens, token_predictions))

            prediction = []
            for token_pred, mapping in zip(wp_preds, offset_mapping.squeeze().tolist()):
            #only predictions on first word pieces are important
                if mapping[0] == 0 and mapping[1] != 0:
                    prediction.append(token_pred[1])
                else:
                    continue

            out_labels = []

            for label in labels:
                try:
                    out_labels.append(IDS2LABELS[label.item()])
                except KeyError:
                    out_labels.append("-")
            
            l = 0
            for i, token in enumerate(tokens):
                if token == "[SEP]":
                    break
                l = i

            tokens_print = "Tokens:       "
            truth_print =  "Ground truth: "
            pred_print =   "Prediction:   "

            for t, o, p in zip(tokens[:l+1], out_labels[:l+1], token_predictions[:l+1]):
                tokens_print += f"{t:<16}"
                truth_print += f"{o:<16}"
                pred_print += f"{p:<16}"

            print(f"\n{tokens_print}")
            print(truth_print)
            print(f"{pred_print}\n")
