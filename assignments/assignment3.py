import numpy as np

X = np.array([(0,0), (1,0), (0,1), (1,1)])
Y = np.array([[0, 0, 0, 1], [0, 1, 1, 1], [0, 1, 1, 0]]) # AND, OR, XOR

# Model - XOR
class ShallowNeuralNetwork():
    def __init__(self, num_input_features, num_hiddens):
        self.num_input_features = num_input_features
        self.num_hiddens = num_hiddens

        # init params
        self.W1 = np.random.normal(size=(self.num_hiddens, self.num_input_features))
        self.b1 = np.random.normal(size=self.num_hiddens)
        self.W2 = np.random.normal(size=self.num_hiddens)
        self.b2 = np.random.normal(size=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        z1 = np.matmul(self.W1, x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.matmul(self.W2, a1) + self.b2
        a2 = self.sigmoid(z2)

        return a2, (z1, a1, z2, a2)

# Train
def train(X, Y, model, lr):
    # init
    dW1 = np.zeros_like(model.W1)
    db1 = np.zeros_like(model.b1)
    dW2 = np.zeros_like(model.W2)
    db2 = np.zeros_like(model.b2)

    n = len(X)
    cost = 0.0

    for x, y in zip(X, Y):
        a2, (z1, a1, z2, _) = model.predict(x)
        if y == 1:
            cost -= np.log(a2)
        else:
            cost -= np.log(1-a2)

        diff = a2 - y
        db2 += diff
        dW2 += a1 * diff

        tmp = (1 - np.power(a1, 2)) * model.W2 * diff
        db1 += tmp
        dW1 += np.matmul(tmp.reshape(-1, 1), np.transpose(x.reshape(-1, 1)))
        # dW1 += np.outer(tmp, x)

    # update
    cost /= n
    model.W1 -= lr * dW1 / n
    model.b1 -= lr * db1 / n
    model.W2 -= lr * dW2 / n
    model.b2 -= lr * db2 / n

    return cost

for epoch in range(100):
    cost = train(X, Y, model, 1.0)
    if epoch % 10 == 0:
        print("Epoch {0} : {1}".format(epoch, cost))

# test
model.predict((1,1))[0].item()
model.predict((1,0))[0].item()
model.predict((0,1))[0].item()
model.predict((0,0))[0].item()