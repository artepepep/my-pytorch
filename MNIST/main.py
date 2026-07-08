from image_utils import load_images, load_labels

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from MLP import MLP
from value import Value
from utils import binary_cross_entropy, zero_grad, step, softmax, one_hot

images = load_images("data/train-images-idx3-ubyte")[:100]
labels = load_labels("data/train-labels-idx1-ubyte")[:100]

model = MLP(784, [16, 10])
learning_rate = 0.01

losses = 10 * [0.0]

for epoch in range(100):
    total_loss = 0

    for inputs, correct_number in zip(images, labels):
        # Forward
        logits = model(inputs)
        preds = softmax(logits)

        target = one_hot(correct_number, 10)

        # Loss
        loss = Value(0)
        for pred, expected in zip(preds, target):
            loss = loss + binary_cross_entropy(expected, pred)

        # Zero grad
        zero_grad(model)

        # Backward
        loss.backward()

        # Learning Step
        step(model, learning_rate)

        total_loss += loss.data
        
    average_loss = total_loss / len(images)
    print(epoch, average_loss)

