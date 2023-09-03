# RTNI2
The second version of Random Tensor Network Integrator ([RTNI](https://motohisafukuda.github.io/RTNI/)) in Python. It can symbolically integrate tensor networks over the Haar-distributed orthogonal and unitary matrices and the real and complex normal Gaussian tensors, even for low dimensions, where the Weingarten functions differ from the ones for high dimensions. One can export tensor networks in the format of [TensorNetwork](https://github.com/google/TensorNetwork).

The latest RTNI2 is compatible with the following versions.
```python
Python 3.11.5
Sympy  1.12
TensorNetwork 0.4.6
Graphviz 0.20.1
```

The original RTNI can integrate tensor networks over random unitary matrices. The Mathematica and Python programs can be found at
[RTNI](https://motohisafukuda.github.io/RTNI/).
This was developped through:<br>
Motohisa Fukuda, Robert König, and Ion Nechita. RTNI - A symbolic integrator for Haar-random tensor networks. \[ [doi](https://doi.org/10.1088/1751-8121/ab434b) | Journal of Physics A: Mathematical and Theoretical, Volume 52, Number 42, 2019 \].
