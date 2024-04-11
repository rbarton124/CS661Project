#include "common.h"
#include "cifar10.h"
#include "test.h"
#include "code_testing.h"

#include <iostream>


int main(int argc, char* argv[]) {

    auto device = get_device();

    if (argc != 5) {
        std::cout << "Usage: ./program <mode:\"cifar10\"/\"imdb\"> <model_path> <test_data_path> <batch_size>\n";

        return 1;
    }

    #ifdef DEBUG
    // test that we can load dataset
    test_load_cifar10("../data/cifar-10-batches-bin/");

    // test that we can load a network and pass data through it
    torch::Tensor example_input = torch::ones({2, 3, 32, 32});
    torch::Tensor example_output = torch::ones({2, 10});

    test_load_compiled_network("../../../python/scripts/dummy_cifar10_network.pt",
                               example_input,
                               example_output);

    test_cifar10_testing("../data/cifar-10-batches-bin/", "../../../python/scripts/dummy_cifar10_network.pt");

    #endif

    std::string mode = std::string(argv[1]);
    std::string model_path = std::string(argv[2]);
    std::string test_data_path = std::string(argv[3]);
    size_t batch_size = stoi(argv[4]);

    Testingresults results;
    if (mode == "cifar10") {
        auto network = load_compiled_network(model_path);
        auto cifar10_testloader = get_CIFAR10_test_dataloader(test_data_path, batch_size);
        TestingResults results = test_model(model, cifar10_testloader);
    } else if (mode == "imdb") {
        throw std::runtime_error("Not implemented");
    } else {
        throw std::runtime_error("Invalid configuration");
    }
    
    display_results(results);

}