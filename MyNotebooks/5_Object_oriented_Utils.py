# Object-Oriented Design for Implementation
# Treating components in deep learning as objects, we can start by defining classes for these objects and their interactions.
# This object-oriented design for implementation will greatly streamline the presentation and you might even want to use it in your projects.

import time
import numpy as np
import torch
from torch import nn

# from d2l import torch as d2l
# import d2l

## Vectorization for speed ##

# First let's see in practice why element-wise operations (and in general vectorization) are useful in machine learning.
# Take two very long arrays and compute an array whose elements are the sum of the elements in the two arrays (element wise).

# define arrays with n = 10000 elements
n = 10000
a = torch.ones(n)
b = torch.ones(n)

# test one: use pure python for loop\n",
start_time = time.time()
c = []
for i in range(n):
    c.append(a[i] + b[i])
end_time = time.time()
print(f"Time taken for pure python loop: {end_time - start_time} seconds")

# compute the sum of the two arrays element-wise\n",
start_time = time.time()
c = a + b
end_time = time.time()
print(f"Time taken for vectorized operation: {end_time - start_time} seconds")

# The vectorized version is much faster!

# test vector-vector multiplication vs matrix-vector multiplication
n = 100000
A = torch.ones((n, n))
b = torch.ones(n)

# vector-vector multiplication
start_time = time.time()
c = torch.mv(A, b)
end_time = time.time()
print(f"Time taken for matrix-vector multiplication: {end_time - start_time} seconds")
# matrix-vector multiplication
start_time = time.time()
c = torch.mm(A, b.reshape((-1, 1)))
end_time = time.time()
print(f"Time taken for matrix-matrix multiplication: {end_time - start_time} seconds")


## Utility Functions ##

### 1: add_to_class

# Notebook readability demands short code fragments, interspersed with explanations, a requirement incompatible with the style of programming common for Python libraries.
# The first utility function allows us to register functions as methods in a class after the class has been created.


def add_to_class(Class):  # @save
    """Register functions as methods in created class."""

    def wrapper(obj):
        setattr(Class, obj.__name__, obj)

    return wrapper


# - add_to_class(Class) returns wrapper.
# - When you use this as a decorator on a function, Python passes the function object as obj to wrapper.
# - `obj.__name__ is the function’s name (string).
# setattr(Class, obj.__name__, obj) attaches that function as a method to Class.


# use example to demonstrate the use of add_to_class
class A:
    def __init__(self):
        self.b = 1


a = A()
print("a = ", a)


@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)


a.do()


### 2: Extend hyperparameters 'callability'


class HyperParameters:  # @save
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=[]):
        "This method is intended to be implemented in a subclass,"
        " but it hasn’t been implemented yet. is a built-in Python "
        " exception used to indicate that a certain method or function"
        " must be overridden in a derived (child) class."
        " By raising it, the base class (probably an abstract class "
        " or interface) forces any subclass to implement its own "
        " version of the loss function"
        raise NotImplemented


# Example use
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=["c"])
        print("self.a =", self.a, "self.b =", self.b)
        print("There is no self.c =", not hasattr(self, "c"))


b = B(a=1, b=2, c=3)
print("b = ", b)


### 3: Plot utilities - interactive plots


class ProgressBoard(d2l.HyperParameters):  # @save
    """The board that plots data points in animation."""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        ls=["-", "--", "-.", ":"],
        colors=["C0", "C1", "C2", "C3"],
        fig=None,
        axes=None,
        figsize=(3.5, 2.5),
        display=True,
    ):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented


board = d2l.ProgressBoard("x")

for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), "sin", every_n=2)
    board.draw(x, np.cos(x), "cos", every_n=10)


# Note: the above code uses the d2l library's ProgressBoard class, rather than the one defined here.
# to see the actual implementation, you would need to refer to the d2l library's source code.

"""Exercise 1"""
# remove the 'save_hyperparameters' statement in the B class
# can you still print self.a and self.b?
