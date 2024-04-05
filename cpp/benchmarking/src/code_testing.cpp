#include "code_testing.h"
#include <cassert>
#include <torch/script.h>

void test_load_cifar10(const std::string& path) {
    auto test_loader = get_CIFAR10_test_dataloader("../data/cifar-10-batches-bin", 32);
}

void test_load_compiled_network(const std::string& path, torch::Tensor example_input, torch::Tensor example_output) {

    auto device = get_device();

    std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    auto network = torch::jit::load(path);

    network.to(device);

    // verify that the network can take the example input and produce
    // an example output of corresponding shape to example output

    example_input = example_input.to(device);

    network.eval();

    torch::NoGradGuard no_grad;

    auto output = network.forward({example_input}).toTensor();

    auto output_sizes = output.sizes().vec();
    auto input_sizes = example_input.sizes().vec();
    auto ref_output_sizes = example_output.sizes().vec();

    auto print_vec = [](auto& vec, const std::string& name) {
        std::cout << name << ": [";
        for(size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if(i < vec.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    };

    if (output_sizes != ref_output_sizes) {
        std::cout << "Failure: The output shape does not match the example output shape." << std::endl;
        std::cout << "Expected shape: " << example_output.sizes() << ", but got: " << output.sizes() << std::endl;
        throw std::runtime_error("Invalid output shape");
    } else {
        print_vec(input_sizes, "example input size");
        print_vec(output_sizes, "output size");
        print_vec(ref_output_sizes, "reference output size");
        std::cout << "Passed shape test!\n";
    }
} 

void test_cifar10_testing(const std::string& cifar10_path, const std::string& model_path) {
    auto dummy_test_dataloader = get_CIFAR10_test_dataloader(cifar10_path, 32);
    auto network = load_compiled_network(model_path);

    TestingResults results = test_model(network, dummy_test_dataloader);

    std::cout << "Dummy model result accuracy: " << results.accuracy << "\n"
              << "Dummy model neg log loss: " << results.neg_log_loss << "\n";
}
