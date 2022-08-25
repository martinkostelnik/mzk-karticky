import typing
import numpy as np
import torch
import os

from dataset import IDS2LABELS
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report


class Trainer:
    def __init__(self, settings: dict, model, tokenizer):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = tokenizer

        # Set training settings
        self.epochs = settings["epochs"]
        self.model = model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=settings["learning_rate"])
        self.max_norm = settings["max_grad_norm"]
        self.num_labels = settings["num_labels"]
        self.output_folder = settings["output_folder"]
        self.load = settings["load"]
        self.debug = settings["debug"]

        if self.load:
            checkpoint = torch.load(os.path.join(self.output_folder, "mzkbert.tar"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optim.load_state_dict(checkpoint["optim_state_dict"])

        # Disable BERT training
        if not settings["bert"]:
            for name, param in self.model.bert.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        if self.debug:
            with open("debug.txt", "a") as f:
                print("\n----------------------------------------------------------------------------------------------------------------------------------\n", file=f)
                print(f"Training on {self.device}", file=f)

    def train(self, train_data_loader, val_data_loader):
        print_steps = 100
        
        # Start training
        for epoch in range(self.epochs):
            if epoch > 0:
                print_steps = 1000

            epoch_acc_train = 0
            epoch_loss_train = 0
            epoch_acc_val = 0
            epoch_loss_val = 0

            train_steps = 0
            val_steps = 0

            steps_loss = 0
            steps_acc = 0

            # Training loop
            self.model.train()
            for i, batch in enumerate(train_data_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs[0], outputs[1]

                epoch_loss_train += loss.item()
                steps_loss += loss.item()

                acc = self.calculate_acc(labels, logits)[0]
                epoch_acc_train += acc
                steps_acc += acc
                
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_steps += 1

                if train_steps % print_steps == 0:
                    print(f"Epoch {epoch+1} | Steps {train_steps} | Loss: {steps_loss / print_steps} | Acc: {steps_acc / print_steps}")
            
                    steps_loss = 0
                    steps_acc = 0

                # Save example input/outputs from first batch
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

            print(f"Epoch {epoch+1} | Loss: {epoch_loss_train / train_steps} | Acc: {epoch_acc_train / train_steps} | Val_Loss: {epoch_loss_val / val_steps} | Val_Acc: {epoch_acc_val / val_steps}")
            
            torch.save({
                        "epoch": self.epochs,
                        "model_state_dict": self.model.state_dict(),
                        "optim_state_dict": self.optim.state_dict(),
            }, os.path.join(self.output_folder, "mzkbert.tar"))

            if self.debug:
                self.print_epoch_example(example_logits, example_labels, example_ids, example_offset_mapping, epoch)

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

    def print_epoch_example(self, logits_, labels_, ids_, offset_mapping_, epoch):
        with open("debug.txt", "a") as f:
            print("\n----------------------------------------------------------------------------------------------------------------------------------\n", file=f)
            print(f"Examples from first batch in epoch {epoch + 1}", file=f)

        for logits, labels, ids, offset_mapping in zip(logits_, labels_, ids_, offset_mapping_):
            active_logits = logits.view(-1, self.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)

            tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
            token_predictions = [IDS2LABELS[i] for i in flattened_predictions.cpu().numpy()]

            out_labels = []
            for label in labels:
                try:
                    out_labels.append(IDS2LABELS[label.item()])
                except KeyError:
                    out_labels.append("-")

            tokens_print = "Tokens:       "
            truth_print =  "Ground truth: "
            pred_print =   "Prediction:   "

            for t, o, p in zip(tokens, out_labels, token_predictions):
                tokens_print += f"{t:<16}"
                truth_print += f"{o:<16}"
                pred_print += f"{p:<16}"

            with open("debug.txt", "a") as f:
                print(f"\n{tokens_print}", file=f)
                print(truth_print, file=f)
                print(f"{pred_print}", file=f)
