from value import Value
import random

class Neuron:
    def __init__(self, n_inputs, bias=0):
        self.weights = [Value(random.uniform(-0.5, 0.5)) for _ in range(n_inputs)]
        self.bias = Value(bias)
    
    def __call__(self, inputs):
        out = self.bias
        for x, w in zip(inputs, self.weights):
            out += x * w
        return out
    
    def parameters(self):
        return self.weights + [self.bias]

# new_neuron = Neuron(5, 2)
# print(new_neuron([0.03, 0.43, 0.31, 0.54, 0.14]))