import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.tuner.tuning import Tuner
import matplotlib.pyplot as plt
import seaborn as sns


# Creation of the model

class BasicLightningTrain(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.learning_rate = 0.1

    def forward(self, input):
        input_to_top_relu = self.w00 * input + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = self.w01 * top_relu_output

        input_to_bottom_relu = self.w10 * input + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = self.w11 * bottom_relu_output

        sum_of_scaled_outputs = (scaled_top_relu_output +
                                 scaled_bottom_relu_output + self.final_bias)

        output = F.relu(sum_of_scaled_outputs)

        return output

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs_i, labels_i = batch
        output_i = self.forward(inputs_i)
        loss = (output_i - labels_i)**2
        return loss


# Implementation of the model

input_doses = torch.linspace(start=0, end=1, steps=11)

print(input_doses)

model = BasicLightningTrain()

output_values = model(input_doses)

sns.set(style='whitegrid')

sns.lineplot(x=input_doses, y=output_values.detach(),
             color='green', linewidth=3)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')

# plt.show()

# plt.savefig('BasicLightning_training.pdf')


# Optimizing the model

inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.])

dataset = TensorDataset(inputs, labels)

dataloader = DataLoader(dataset)

model = BasicLightningTrain()

trainer = L.Trainer(max_epochs=34)

tuner = Tuner(trainer)


lr_find_results = tuner.lr_find(model=model,
                                train_dataloaders=dataloader,
                                min_lr=0.001,
                                max_lr=1.0,
                                early_stop_threshold=None)


new_lr = lr_find_results.suggestion()

print(f"lr_find() suggests {new_lr:.5f} for the learning rate.")

model.learning_rate = 0.1

trainer.fit(model, train_dataloaders=dataloader)

print(model.final_bias.data)


# Implementation of the model adjusted

output_values = model(input_doses)

sns.set(style='whitegrid')

sns.lineplot(x=input_doses, y=output_values.detach(),
             color='green', linewidth=3)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')

# plt.show()
