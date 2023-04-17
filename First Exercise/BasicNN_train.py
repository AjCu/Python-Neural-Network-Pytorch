import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns

import matplotlib.pyplot as plt

## Creation of the model

class BasicNN_train(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, input):
        input_to_top_relu = self.w00 * input + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = self.w01 * top_relu_output

        input_to_bottom_relu = self.w10 * input + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = self.w11 * bottom_relu_output

        sum_of_scaled_outputs = scaled_top_relu_output + \
            scaled_bottom_relu_output + self.final_bias

        output = F.relu(sum_of_scaled_outputs)

        return output


## Implementation of the model

input_doses = torch.linspace(start=0, end=1, steps=11)

print(input_doses)

model = BasicNN_train()

output_values = model(input_doses)

sns.set(style='whitegrid')

sns.lineplot(x=input_doses, y=output_values.detach(),
             color='green', linewidth=3)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')

plt.show()

# plt.savefig('BasicNN_training.pdf')


# Optimizing the model

inputs = torch.tensor([0., 0.5, 1.])

labels = torch.tensor([0., 1., 0.])

optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Final bias before optimization: ", str(model.final_bias.data), " \n")


for epoch in range(100):

    total_loss = 0
    
    for iteration in range(len(inputs)):

        input_i = inputs[iteration]
        label_i = labels[iteration]

        output_i = model(input_i)

        loss = (output_i - label_i) ** 2

        loss.backward()

        total_loss += float(loss)

    if (total_loss < 0.0001):
        print("Num steps: ", str(epoch))
        print("Total loss: ", str(total_loss))
        break

    optimizer.step()
    optimizer.zero_grad()

    print("Step: ", str(epoch), " Final Bias: ", str(model.final_bias.data), " \n")

print("Final bias after optimization: ", str(model.final_bias.data))

## saving results of the optimization

output_values = model(input_doses)

sns.set(style="whitegrid")


sns.lineplot(x=input_doses, 
             y=output_values.detach(),
             color='green', 
             linewidth=3)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')

plt.show()
##plt.savefig('BasicNN_optimized.pdf')
