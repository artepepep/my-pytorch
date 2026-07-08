import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from MLP import MLP

model = MLP(2, [4, 3, 1])

params = model.parameters()
model_path = Path(__file__).with_name("model.json")

with open(model_path, "r") as file:
    model_data = json.load(file)

for p, saved_value in zip(params, model_data):
    p.data = saved_value


def predict(inputs):
    y_pred = model(inputs)[0].sigmoid()

    print("probability:", y_pred.data)

    if y_pred.data >= 0.5:
        return 1
    else:
        return 0


print("0 xor 0 =", predict([0, 0]))
print("0 xor 1 =", predict([0, 1]))
print("1 xor 0 =", predict([1, 0]))
print("1 xor 1 =", predict([1, 1]))
