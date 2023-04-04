# NN-from-scratch
My attempt at the implementation of a neural network using numpy.

## Try

``` python
import numpy as np            
X = [35, 83, 17, 14, 90, 90, 45, 52, 60, 83, 73, 97, 70, 13, 73, 69, 6, 6, 6, 8]
y = np.array([1, 0, 0, 0])

nn = NN(X, y, 6)

print("Before Train:", nn.getArgMax())
nn.train()
print("After Train:", nn.getArgMax())
```