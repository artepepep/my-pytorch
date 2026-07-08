from value import Value
from layer import Layer

class MLP:
    def __init__(self, input, layers):
        self.sizes = [input] + layers
        self.input = Value(input)
        self.layers = [Layer(self.sizes[index], self.sizes[index + 1]) for index in range(len(self.sizes) - 1)]

    def __call__(self, inputs):
        x = inputs

        for index, layer in enumerate(self.layers):
            x = layer(x)

            if index != len(self.layers) - 1:
                x = [v.relu() for v in x]

        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]