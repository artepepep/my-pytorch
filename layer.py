from neuron import Neuron

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]
    
    def __call__(self, inputs):
        return [neuron(inputs) for neuron in self.neurons]
    
    def __repr__(self):
        return f"Layer({self.n_inputs} -> {self.n_outputs})"
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]