from neuron import Neuron

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]
    
    def __call__(self, inputs):
        return [neuron(inputs) for neuron in self.neurons]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

first_layer = Layer(3, 2)
print(first_layer([0.5, 0.42, 0.14]))