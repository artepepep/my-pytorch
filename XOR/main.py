import json
import sys
from pathlib import Path

import pandas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from MLP import MLP
from utils import binary_cross_entropy, zero_grad, step

info = pandas.read_csv(PROJECT_ROOT / "data" / "Xor_Dataset.csv")

dataset = []

for _, row in info.iterrows():
    inputs = [row["X"], row["Y"]]
    answer = row["Z"]
    dataset.append((inputs, answer))

learning_rate = 0.01

model = MLP(2, [4, 3, 1])

for epoch in range(25):
    total_loss = 0

    for inputs, correct_answer in dataset:
        # Forward
        pred = model(inputs)[0].sigmoid()

        # Loss
        loss = binary_cross_entropy(correct_answer, pred)

        # Gradients to zero
        zero_grad(model)

        # Backward
        loss.backward()

        # Learning step 
        step(model, learning_rate)

        total_loss += loss.data

    average_loss = total_loss / len(dataset)
    print(epoch, average_loss)

model_data = [p.data for p in model.parameters()]
model_path = Path(__file__).with_name("model.json")

with open(model_path, "w") as file:
    json.dump(model_data, file)
