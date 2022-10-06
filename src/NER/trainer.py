import typing
import numpy as np
import torch
import os

from helper import calculate_acc


class Trainer:
    def __init__(self, settings: dict, model, tokenizer):
        self.tokenizer = tokenizer

        # Set training settings
        self.epochs = settings["epochs"]
        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), lr=settings["learning_rate"])
        self.max_norm = settings["max_grad_norm"]

        # Set misc settings
        self.num_labels = model.config.num_labels
        self.output_folder = settings["output_folder"]

        # Disable BERT training
        if not settings["bert"]:
            for _, param in self.model.bert.named_parameters():
                param.requires_grad = False

    def train_step(self, batch):
        loss, logits = self.forward(batch)

        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item(), logits.detach()

    def test_step(self, batch):
        with torch.no_grad():
            loss, logits = self.forward(batch)

        return loss.item(), logits.detach()

    def forward(self, batch):
        device = self.model.get_device()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        loss, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return loss, logits[0]

    def train(self, train_data_loader, val_data_loader, view_step=1000):
        # Start training
        for epoch in range(self.epochs):
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
                loss, logits = self.train_step(batch)

                epoch_loss_train += loss
                steps_loss += loss

                acc = calculate_acc(batch["labels"], logits, self.model.num_labels)[0]
                epoch_acc_train += acc
                steps_acc += acc

                train_steps += 1

                if train_steps % view_step == 0:
                    print(f"Epoch {epoch+1} | Steps {train_steps} | Loss: {steps_loss / view_step} | Acc: {steps_acc / view_step}")
            
                    steps_loss = 0
                    steps_acc = 0

            # Validation loop
            self.model.eval()
            for batch in val_data_loader:
                loss, logits = self.test_step(batch)

                epoch_loss_val += loss
                epoch_acc_val += calculate_acc(batch["labels"], logits, self.model.num_labels)[0]

                val_steps += 1

            print(f"Epoch {epoch+1} | Loss: {epoch_loss_train / train_steps} | Acc: {epoch_acc_train / train_steps} | Val_Loss: {epoch_loss_val / val_steps} | Val_Acc: {epoch_acc_val / val_steps}")
            self.model.save(os.path.join(self.output_folder, f"checkpoint_{epoch+1:03d}.pth"))
