import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns


# Creation of the model

class BasicLightning(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

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


# Implementation of the model

input_doses = torch.linspace(start=0, end=1, steps=11)

print(input_doses)

model = BasicLightning()

output_values = model(input_doses)

sns.set(style='whitegrid')

sns.lineplot(x=input_doses, y=output_values, color='green', linewidth=3)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')

plt.show()