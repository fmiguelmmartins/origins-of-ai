"""
First NN from scratch using:
Mean Absolute Error
"""

import numpy as np


inputs = [1, 2, 3, 4, 5]
targets = [23, 26, 29, 32, 35]

w = 0.1
b = 0
learning_rate = 0.01
epochs = 6000


def predict(i):
    return w * i + b


for epoch in range(epochs):

    pred = [predict(i) for i in inputs]

    # Cost, loss or error = the same thing!
    costs = [t - p for p, t in zip(pred, targets)]

    # Sign function
    sgn_costs = [1 if c > 0 else -1 for c in costs]

    cost_average = sum(abs(e) for e in costs) / len(costs)
    print(f"Epoch: {epoch}, Bias: {b} Weight: {w:.2f} Cost:{cost_average:.2f}")

    w += learning_rate * sum(e * x for e, x in zip(sgn_costs, inputs)) / len(inputs)
    b += learning_rate * sum(sgn_costs) / len(inputs)


# test the network
test_inputs = [6, 7]
test_targets = [38, 41]

pred = [predict(i) for i in test_inputs]

for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"input:{i}, target:{t}, pred: {p}")
