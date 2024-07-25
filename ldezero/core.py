import numpy as np


class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # ------ recurrent implementation ------
        # f = self.creator        # 1. get func
        # if f is not None:
        #     x = f.input         # 2. get func's input
        #     x.grad = f.backward(self.grad)      # call func's backward
        #     x.backward()        # 调用自己前面那个变量的backward方法（递归）
        
        # ------ for loop implementation -----
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()                 # 1. get func
            x, y = f.input, f.output        # 2. get func's input
            x.grad = f.backward(y.grad)     # 3. backward call backward

            if x.creator is not None:
                funcs.append(x.creator)     # 将前一个函数添加到list中


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)

        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)    # make output variable save creator information
        
        self.inputs = inputs          # save the variable of input
        self.outputs = outputs        # alse save output variable

        return outputs
    
    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2

        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy

        return gx


def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)

        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy

        return gx


def exp(x):
    return Exp()(x)


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1

        return (y,)


if __name__ == '__main__':
    # A = Square()
    # B = Exp()
    # C = Square()

    x = Variable(np.array(0.5))
    # x = Variable(None)

    # x = Variable(1.0)           # TypeError
    # --------------------------
    # a = square(x)
    # b = exp(a)
    # y = square(b)
    # -->
    y = square(exp(square(x)))
    # --------------------------
    # print(f'{y=}')

    # assert y.creator == C

    # y.grad = np.array(1.0)

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



