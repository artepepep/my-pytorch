import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1

        for node in reversed(topo):
            node._backward()

    def relu(self):
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Value(other) / self

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            _children=(self, other),
            _op="*"
        )

        def _backward():
            other.grad += self.data * out.grad
            self.grad += other.data * out.grad

        out._backward = _backward # сохраняем функцию внутри out
        return out
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            _children=(self, other),
            _op="+"
        )

        def _backward():
            other.grad += out.grad
            self.grad += out.grad

        out._backward = _backward # сохраняем функцию внутри out
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data - other.data,
            _children=(self, other),
            _op="-"
        )

        def _backward():
            other.grad -= out.grad
            self.grad += out.grad

        out._backward = _backward # сохраняем функцию внутри out
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))

        out = Value(
            self.data ** other,
            _children=(self,),
            _op=f"**{other}"
        )

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self
    
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
