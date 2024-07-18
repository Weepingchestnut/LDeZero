import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        f = self.creator        # 1. get func
        if f is not None:
            x = f.input         # 2. get func's input
            x.grad = f.backward(self.grad)      # call func's backward
            x.backward()        # 调用自己前面那个变量的backward方法（递归）


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)

        output = Variable(y)
        output.set_creator(self)    # make output variable save creator information
        
        self.input = input          # save the variable of input
        self.output = output        # alse save output variable

        return output
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2

        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy

        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)

        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy

        return gx


if __name__ == '__main__':
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    # print(f'{y=}')

    assert y.creator == C

    y.grad = np.array(1.0)

    y.backward()

    # ----------------------------
    # C = y.creator
    # b = C.input
    # b.grad = C.backward(y.grad)

    # B = b.creator
    # a = B.input
    # a.grad = B.backward(b.grad)

    # A = a.creator
    # x = A.input
    # x.grad = A.backward(a.grad)

    # ---------------------------
    # b.grad = C.backward(y.grad)
    # a.grad = B.backward(b.grad)
    # x.grad = A.backward(a.grad)

    print(f'{x.grad=}')



