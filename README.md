# Neural quantum solver for excited states of trapped ion quantum computing

This repository contains the code and data for the paper [Solving excited states for long-range interacting trapped ions with neural networks](https://arxiv.org/abs/2506.08594).

## Organization

- `code/Jmat/code/Ising2d_power_law.py`: Python code to generate the interactive Hamiltonian (the J_ij matrix) for the trapped ion model
- `code/rbm.cc`: Optimized C++ code for the neural quantum solver with and without the matrix-free method. This is the code that generated the data in the paper
- `data/energy`: Energy spectrum generated at each step of the training process
- `data/corrfcn`: Data for the correlation functions
- `data/rbm-weights`: Weights of the RBM neural networks at the end of the training process

## Building and running the code

To build the code, change directory to `code` and simply run
```
./build.sh [opt]
```
Without `opt`, the default is to generate debug builds which are unoptimized and contain all the debugging information. With `opt`, the compiler will be instructed to generate optimized code without debugging symbols in the executable.

To run the code, use the script `run.sh`. For instance,
```
./run.sh ./rbm-matfree-N50-M150-K2-opt -c 128 --model trappedion -h 0.595 -n 200 -m 200 --tol 1e-6 --reglam-init 40 --reglam-cutoff 1 --minres-rtol 1e-9 --correlation-data ioncorrN50M150K2h0.595.mat --dump-rbm ionh0.595
```
will perform a simulation of 50 trapped ions using RBM states with 150 hidden units, and solve for the two lowest energy states in the spectrum, using the optimized matrix-free method. RBM weights are correlation function data will be stored to the specified MATLAB .mat files. For a complete list of supported command line parameters, do
```
./rbm-matfree-N50-M150-K2-opt -?
```

To build additional executables for different combinations of N, M, and K, change the line in the build script
```
PARAMS=("8 16 2" ...)
```
to the parameters you want. The format is "N M K", in this order. For instance, to build the executable for an RBM solver with N=10, M=20, K=3, do
```
PARAMS=("10 20 3")
```

If the specified N is not in the list of pre-built J matrices, you must generate the corresponding J matrix using the code supplied under `code/Jmat/code`. Simply specify N as the parameter to the script, for instance,
```
python Ising2d_power_law.py 10
```
if you want to generate the J matrix for the trapped ion model with N=10 ions.

### Data format

Data under `data/energy` are text files listing the energy spectrum at each step of the training processes. Data under `data/corrfcn` and `data/rbm-weights` are matrices stored in the MATLAB .mat file (version 5), which can be processed using standard tools such as python's scipy package.

## Citing our paper
```
@misc{ma2025solvingexcitedstateslongrange,
      title={Solving excited states for long-range interacting trapped ions with neural networks},
      author={Yixuan Ma and Chang Liu and Weikang Li and Shun-Yao Zhang and L. -M. Duan and Yukai Wu and Dong-Ling Deng},
      year={2025},
      eprint={2506.08594},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2506.08594},
}
```
