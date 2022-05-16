import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
# visualization
import matplotlib.pyplot as plt
# import seaborn as sns


# Model
class MLP(nn.Module):
    def __init__(self, num_layers=5):
        super(MLP, self).__init__()

        self.num_layers = num_layers

        self.in_dim = 28*28 # MNIST
        self.out_dim = 10 # 0 ~ 9

        self.n_nodes = [self.in_dim, 512, 256, 128, 64, self.out_dim]

        self.linears = nn.ModuleList()
        for i in range(self.num_layers):
            if i == self.num_layers - 1: # last layer
                n_tmp = self.out_dim
            else:
                n_tmp = self.n_nodes[i+1]
            self.linears.append(nn.Linear(self.n_nodes[i], n_tmp))

        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        for i, l in enumerate(self.linears):
            if i == self.num_layers - 1:
                logit = l(x)
            else:
                x = self.relu(l(x))

        return logit


if __name__=="__main__":
    # Data Load
    data_path = "./dataset"
    train_data = MNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
    test_data = MNIST(data_path, train=False, download=True, transform=transforms.ToTensor())

    batch_size = 12
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)


    n_layers = 5
    acc_list = []

    for n in range(n_layers):
        print("Num of Layers : ", n+1)
        model = MLP(num_layers=n+1)
        print(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Training
        for epoch in tqdm(range(10)):
            
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data # [inputs, labels]

                optimizer.zero_grad() # init gradients

                outputs = model(inputs) # output
                loss = criterion(outputs, labels) # loss
                loss.backward() # backpropagate
                optimizer.step() # update

        # Testing
        n_predict = 0
        n_correct = 0

        for data in test_loader:
            inputs, labels = data # [inputs, labels]
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            n_predict += len(predicted)
            n_correct += (labels == predicted).sum()

        acc_list.append(n_correct/n_predict)


    # Visualization
    x = range(n_layers)
    y = acc_list[:]
    num_layers = ["1", "2", "3", "4", "5"]
    # colors = sns.color_palette("Set2",len(x))

    # plt.bar(x, y, width=0.6, tick_label=num_layers, color=colors)
    plt.plot(x, y, ".-", color="r")
    plt.xticks(x, num_layers)
    plt.title("Multi-Layer Perceptron")
    plt.xlabel("Number of Layers")
    plt.ylabel("Accuracy")
    for i, v in enumerate(x):
        plt.text(v, y[i], str(round(y[i].item(), 3)), horizontalalignment='center', verticalalignment='bottom')

    plt.savefig("./result_plot.png")
    plt.show()
