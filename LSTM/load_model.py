import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

only_last = False

class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM1, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)


        # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0)) # x is the input
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        if only_last:
            out = out[:, -1, :]

        out = self.fc(out)

        if not only_last:
            out = torch.permute(out, (0,2,1))

        return out

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

test_input = torch.tensor(x_test, dtype=torch.float32)
test_input = torch.reshape(test_input, (1, x_test.shape[0], x_test.shape[1]))

test_output = torch.tensor(y_test, dtype=torch.float32)
test_output = torch.reshape(test_output, (1, y_test.shape[0]))

best_model = torch.load('RestCoPas200/New/best.ckpt')
outputs = best_model(test_input)
_, predicted = torch.max(outputs.data, 1)

# Save for plotting graphs

time = np.arange(test_output.shape[1])
np.save('timetest.npy',time)
np.save('expected.npy', test_output)
np.save('predicted.npy', predicted)
plt.plot(time, test_output[0], label='expected')
plt.plot(time, predicted[0], label='predicted')
plt.legend()
plt.show()
