import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Not predict the only last behavior of the sample :
only_last = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import Data
x = np.load("intensities_long.npy") #(7200,3)
y = np.load("behavior_long.npy") #(7200,1)

sequence_length = x.shape[0]
stop = int(0.8*sequence_length)
print(stop) #5760

# Create a gap
decalage = 0
xdec = np.zeros((x.shape[0]-decalage, x.shape[1]))
ydec = np.zeros((y.shape[0]-decalage))

for i in range (ydec.shape[0]):
    ydec[i]=y[i+decalage]
    xdec[i]=x[i]
    #xdec[i,0]=0  # to not consider the intensity of region low
    #xdec[i,1]=0  # not consider the region mid


print("je suis les 10 premiers de xdec \n", xdec[:10])
print(xdec.shape, ydec.shape)

# Create the different datasets
x_train=xdec[:stop,:]
x_test=xdec[stop:,:]

y_train=ydec[:stop]
y_test=ydec[stop:]


np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

# Compute the test shape

test_input=torch.tensor(x_test, dtype=torch.float32)
test_input=torch.reshape(test_input, (1, x_test.shape[0], x_test.shape[1]))
#print(test_input.shape) #torch.Size([1 , 1440, 3])
test_output=torch.tensor(y_test, dtype=torch.float32)
test_output=torch.reshape(test_output, (1, y_test.shape[0]))
#print(test_output.shape) #torch.Size([1, 1440])


# Hyper-parameters

num_classes = 4   # number of classes to predict
num_epochs = 100
num_steps = 100
learning_rate = 0.001
input_size = x.shape[1]   # nb of features = regions
hidden_size = 128
num_layers = 2    # 2 stacked layers
best_acc = 0

acc_tab = []

# Fully connected neural network

class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM1, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # batch_first = True so the data have to be with shape (batch_size, sequence_length, nb_features=input_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        # to have the desired shape output

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
            out = torch.permute(out, (0,2,1)) # need to permute so CrossEntropyLoss can interpret it

        return out

model = LSTM1(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)

# Loss and optimizer

if only_last:
    criterion = nn.CrossEntropyLoss()
if not only_last:
    criterion = nn.CrossEntropyLoss(reduction='none')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

totalloss = []

# Training and test loop

for epoch in range(num_epochs):
    i = 0

    # Train the model
    for step in range(num_steps):
        x_train_samples = []
        y_train_samples = []

        #create samples of size pas

        pas = 200

        indextrain = np.arange(stop-pas)
        randomindextrain = np.random.choice(indextrain,16)

        for nb in randomindextrain :
            x_train_samples.append(x_train[nb:nb+pas])
            y_train_samples.append(y_train[nb:nb+pas])


        x_train_samples = np.array(x_train_samples) # (16, 200, 3)
        y_train_samples = np.array(y_train_samples) # (16, 200, 3)

        train_input = torch.tensor(x_train_samples, dtype=torch.float32)
        #print(train_input.shape) #torch.Size([16, 200, 3])
        train_output = torch.tensor(y_train_samples, dtype=torch.float32)
        #print(train_output.shape) #torch.Size([16, 200])


        if only_last:
            train_output = train_output[:,-1].long()
            test_output = test_output[:,-1].long()
            print(train_output.dtype)
        if not only_last:
            train_output = train_output.long()
            test_output = test_output.long()

        # Forward pass
        outputs = model(train_input)
        loss = criterion(outputs, train_output)
        if not only_last:
            weights = torch.linspace(0.01,1,train_input.shape[1])
            loss = loss*weights
            loss = loss.mean()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], for step [{step+1}/{num_steps}], Loss: {loss.item():.4f}')
            totalloss.append(loss.detach().numpy())
        i+=1


    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        outputs = model(test_input)

        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples = test_output.shape[1]*test_output.shape[0]
        n_correct = (predicted == test_output).sum()

        acc = 100.0 * n_correct / n_samples

        loss = criterion(outputs, test_output)
        if not only_last:
            weights = torch.linspace(0.01,1,test_input.shape[1])
            loss = loss*weights
            loss = loss.mean()

        print(loss)
        print(f'Accuracy of the network on the {n_samples} test images: {acc} %')

        # Save accuracy to plot it later
        acc_tab.append(acc)

        # Save the best model if need to use it later
        if acc > best_acc :
            torch.save(model, 'best.ckpt')
            best_acc = acc

best_model = torch.load('best.ckpt')
outputs = best_model(test_input)
_, predicted = torch.max(outputs.data, 1)
print('je suis best acc : ', best_acc)

acc_tab = np.array(acc_tab)

# Save Loss for plotting graphs

totalloss = np.array(totalloss)
np.save('totalloss.npy', totalloss)
time = np.arange(num_epochs)
np.save('timetrain.npy', time)
np.save('accuracy.npy', acc_tab)

# Save Test for plotting graphs

time = np.arange(test_output.shape[1])
np.save('timetest.npy',time)
np.save('expected.npy', test_output)
np.save('predicted.npy', predicted)
plt.plot(time, test_output[0], label='expected')
plt.plot(time, predicted[0], label='predicted')
plt.legend()
plt.show()
