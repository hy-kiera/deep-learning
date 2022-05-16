import numpy as np
import matplotlib.pyplot as plt


# Data
X = np.array([(0,0), (1,0), (0,1), (1,1)])
Y = np.array([[0, 0, 0, 1], [0, 1, 1, 1], [0, 1, 1, 0]]) # AND, OR, XOR
Y_name = np.array(["AND", "OR", "XOR"])

lr = np.array([0.1, 0.01, 0.001])

# Model
class LogitsticRegression():
    def __init__(self):
        self.W = np.random.normal(size=2)
        self.b = np.random.normal(size=1)

    def sigmoid(slef, z):
        return 1 / (1 - np.exp(-z))

    def predict(self, x):
        z = np.inner(self.W, x) + self.b[0]
        return self.sigmoid(z)

# train
def train(X, Y, model, lr = 0.01):
    dW = np.zeros((2))
    db = np.zeros((1))
    m = X.shape[0]
    cost = 0.0

    for x, y in zip(X, Y):
        y_hat = model.predict(x)
        # L(y, y_hat) = -y * log(y_hat) - (1-y) * log(1-y_hat)
        if y == 1:
            cost -= np.log(y_hat)
        else:
            cost -= np.log(1-y_hat)

        dW += (y_hat - y) * x
        db += (y_hat - y)

    cost /= m
    model.W -= lr * dW / m
    model.b -= lr * db / m

    return cost

# train
loss = []
for i in range(len(Y)):
    for j in range(len(lr)):
        print("Data : {0} with lr({1})".format(Y_name[i], lr[j]))
        model = LogitsticRegression()
        for epoch in range(10000):
            cost = train(X, Y[i], model, lr[i])
            loss.append(cost)
            if epoch % 3000 == 0:
                print(epoch, cost)

        # predict results
        for k in range(len(X)):
            print("{0} : {1}".format(X[k], model.predict(X[k])))
        print("="*40)
    print()

# Loss plot
x = np.arange(10000)
y = np.array(loss).reshape((len(Y)*len(lr), -1))

fig = plt.figure(figsize=(15, 5))
for i in range(len(Y)):
    ax = fig.add_subplot(1, 3, i+1)
    ax.plot(x, y[i*3], label="0.1", c="b")
    ax.plot(x, y[i*3+1], label="0.01", c="g")
    ax.plot(x, y[i*3+2], label="0.001", c="r")
    ax.legend()
    ax.set_title(Y_name[i])
    ax.set_ylim([0, np.max(y)+0.2])
    ax.set_xlabel("Epoch")
    ax.set_ylbael("Loss")

plt.show()