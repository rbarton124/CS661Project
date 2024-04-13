
# Download libtorch info
https://pytorch.org/get-started/locally/

Download libtorch from this website. For this library, I am using
Stable - Linux - Libtorch - C++/Java - CUDA 12.1 - cxx11 ABI

Put it in a folder of your choice. 

# Including libtorch

In the CMakeLists file, set the CMAKE_PREFIX_PATH to the libtorch directory.

# Rough program flow
The rough structure of the files look like this:
* cifar10 - responsible for loading cifar10 dataset, with some constants defined in the .h
* network - responsible for loading pre-compiled networks w/ python jit
* code_testing - responsible for correctness tests & just making sure things don't just crash
* common - just some common utilities
* test - used for inference, you'd need to pass the model and the test dataloader
* main - meant to put high level code logic for what you wanna do

# Relevant paths/directories may have to change
* In main, you will see that there are 2 paths of concern
    * The first is the path directory of the data files. This should be automatically set to the right one at ../data/cifar-10-batches-bin
    * The second is the directory of the compiled model. For the dummy model, it is located at ../../../python/scripts/dummy_cifar10_network.pt

# Todos:

* ~~Comments~~
* Turn it into an executable
* C++ profiling equivalent of tensorboard 

## Main API breakdown

After building, the following should become available
./test_inference <datatype> <model data path> <relative model path>

<datatype> - currently only supports "cifar10"


