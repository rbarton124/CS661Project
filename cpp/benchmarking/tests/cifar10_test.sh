cd ..
cd build
cmake ..
make

./torch_cpp_benchmark cifar10 ../../../python/scripts/dummy_cifar10_network.pt ../data/cifar-10-batches-bin 32