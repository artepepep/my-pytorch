from layer import Layer
import pandas
import json
from utils import binary_cross_entropy

info = pandas.read_csv('/Users/artemvolos/nn-learning/my-pytorch/data/Xor_Dataset.csv')

dataset = []

for _, row in info.iterrows():
    inputs = [row["X"], row["Y"]]
    answer = row["Z"]
    dataset.append((inputs, answer))

learning_rate = 0.01
layer1 = Layer(2, 4)
layer2 = Layer(4, 1)

params = (
    layer1.parameters()
    + layer2.parameters()
)

for epoch in range(25):
    total_loss = 0

    for inputs, correct_answer in dataset:
        # Forward
        out1 = layer1(inputs)
        out1 = [x.relu() for x in out1]

        out2 = layer2(out1)
        y_pred = out2[0].sigmoid()

        # Loss
        loss = binary_cross_entropy(correct_answer, y_pred)
        total_loss += loss.data

        # Gradients to zero
        for p in params:
            p.grad = 0

        # Backward
        loss.backward()

        for p in params:
            p.data = p.data - learning_rate * p.grad

    average_loss = total_loss / len(dataset)
    print(epoch, average_loss)

model_data = [p.data for p in params]

with open("model.json", "w") as file:
    json.dump(model_data, file)