import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data - other.data,
            _children=(self, other),
            _op="-"
        )

        def _backward():
            other.grad -= out.grad
            self.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.grad * out.grad
            other.grad += self.grad * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(
            self.data @ other.data,
            (self, other),
            "@"
        )

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for child in node._children:
                    build_topo(child)

                topo.append(node)

        self.grad = 1
        for t in reversed(topo):
            t._backward()

    def sum(self):
        return self.data.sum()

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self, ), 'relu')

        def _backward():
            self.grad = (self.data > 0) * out.grad
        
        out._backward = _backward
        return out

tensor1 = Tensor([1, -5, 12, -10])
tensor2 = tensor1.relu()