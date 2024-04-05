#include "common.h"
#include "cifar10.h"
#include "test.h"
#include "code_testing.h"

#include <iostream>


int main(int argc, char* argv[]) {

    auto device = get_device();

    #ifdef DEBUG
    // test that we can load dataset
    test_load_cifar10("../data/cifar-10-batches-bin/");

    // test that we can load a network and pass data through it
    torch::Tensor example_input = torch::ones({2, 3, 32, 32});
    torch::Tensor example_output = torch::ones({2, 10});

    test_load_compiled_network("../../../python/scripts/dummy_cifar10_network.pt",
                               example_input,
                               example_output);

    test_testing("../data/cifar-10-batches-bin/", "../../../python/scripts/dummy_cifar10_network.pt");

    #endif
}