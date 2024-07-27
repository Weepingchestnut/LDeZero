from typing import Tuple
import numpy as np


class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
    
    def cleargrad(self):
        self.grad = None        # init grad
    
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
            # x, y = f.input, f.output        # 2. get func's input
            # x.grad = f.backward(y.grad)     # 3. backward call backward
            # --> 支持可变长参数
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, Tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                # x.grad = gx
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)     # 将前一个函数添加到list中


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]

        ys = self.forward(*xs)
        if not isinstance(ys, Tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)    # make output variable save creator information
        
        self.inputs = inputs          # save the variable of input
        self.outputs = outputs        # alse save output variable

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2

        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
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
    def forward(self, x0, x1):
        y = x0 + x1

        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


if __name__ == '__main__':
    
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print(f'{x.grad}')

    x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    print(f'{x.grad}')



