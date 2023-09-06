# PyRTNI2
The second version of Random Tensor Network Integrator ([RTNI](https://motohisafukuda.github.io/RTNI/)) in Python. It can symbolically integrate tensor networks over the Haar-distributed orthogonal and unitary matrices and the real and complex normal Gaussian tensors, even for low dimensions, where the Weingarten functions differ from the ones for high dimensions. One can export tensor networks in the format of [TensorNetwork](https://github.com/google/TensorNetwork).

The latest PyRTNI2 is compatible with the following versions.
```python
Python 3.11.5
Sympy  1.12
TensorNetwork 0.4.6
Graphviz 0.20.1
```

The paper is at the arxiv [arXiv:2309.01167 [physics.comp-ph]]([https://doi.org/10.1088/1751-8121/ab434b](https://arxiv.org/abs/2309.01167)) and please cite it as
```
@misc{fukuda2023symbolically,
      title={Symbolically integrating tensor networks over various random tensors -- the second version of Python RTNI}, 
      author={Motohisa Fukuda},
      year={2023},
      eprint={2309.01167},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph}
}
```

The original RTNI can integrate tensor networks over random unitary matrices. The Mathematica and Python programs can be found at another repository:<br>
[RTNI](https://github.com/MotohisaFukuda/PyRTNI2)).
This was developped through:<br>
Motohisa Fukuda, Robert König, and Ion Nechita. RTNI - A symbolic integrator for Haar-random tensor networks.<br>
Journal of Physics A: Mathematical and Theoretical, Volume 52, Number 42, 2019.<br>
[This original paper](https://doi.org/10.1088/1751-8121/ab434b) can be cited as
```
@article{fukuda2019rtni,
  title={RTNI—A symbolic integrator for Haar-random tensor networks},
  author={Fukuda, Motohisa and Koenig, Robert and Nechita, Ion},
  journal={Journal of Physics A: Mathematical and Theoretical},
  volume={52},
  number={42},
  pages={425303},
  year={2019},
  publisher={IOP Publishing}
}
```
