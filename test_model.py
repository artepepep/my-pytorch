import json
from layer import Layer

layer1 = Layer(2, 4)
layer2 = Layer(4, 1)

params = layer1.parameters() + layer2.parameters()

with open("model.json", "r") as file:
    model_data = json.load(file)

for p, saved_value in zip(params, model_data):
    p.data = saved_value


def predict(inputs):
    out1 = layer1(inputs)
    out1 = [x.relu() for x in out1]

    out2 = layer2(out1)
    y_pred = out2[0].sigmoid()

    print("probability:", y_pred.data)

    if y_pred.data >= 0.5:
        return 1
    else:
        return 0


print("0 xor 0 =", predict([0, 0]))
print("0 xor 1 =", predict([0, 1]))
print("1 xor 0 =", predict([1, 0]))
print("1 xor 1 =", predict([1, 1]))