#pragma once

#include "common.h"

#include <string>
#include <fstream>
#include <cstddef>
#include <torch/torch.h>

namespace CIFAR10_CONSTANTS {
    constexpr uint32_t TRAIN_SIZE = 50000;
    constexpr uint32_t TEST_SIZE = 10000;
    constexpr uint32_t SIZE_PER_BATCH = 10000;
    constexpr uint32_t IMAGE_ROWS = 32;
    constexpr uint32_t IMAGE_COLS = 32;
    constexpr uint32_t BYTES_PER_ROW = 3073;
    constexpr uint32_t BYTES_PER_CHANNEL_PER_ROW = 1024;
    constexpr uint32_t BYTES_PER_BATCH_FILE = BYTES_PER_ROW * SIZE_PER_BATCH;
    constexpr uint32_t CHANNELS = 3;
    constexpr uint32_t NUM_CLASSES = 10;
    
    const std::vector<std::string> TRAIN_LOCATIONS = {
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"
    };

    const std::string TEST_LOCATION = "test_batch.bin";
};

class CIFAR10Test: public torch::data::datasets::Dataset<CIFAR10Test> {
public:
    explicit CIFAR10Test(const std::string& root);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

    const torch::Tensor& get_images() const;
    const torch::Tensor& get_targets() const;

private:
    static std::pair<torch::Tensor, torch::Tensor> load(const std::string& root);
    torch::Tensor images, targets;
};

inline auto get_CIFAR10_test_dataloader(const std::string& root, size_t batch_size) {
    auto test_dataset = CIFAR10Test(root).map(torch::data::transforms::Stack<>());

    #ifdef DEBUG
    assert(*(test_dataset.size()) == CIFAR10_CONSTANTS::TEST_SIZE);
    #endif
    
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size
    );

    return test_loader;
}