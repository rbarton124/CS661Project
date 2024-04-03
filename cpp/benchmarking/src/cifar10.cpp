#include "cifar10.h"

CIFAR10Test::CIFAR10Test(const std::string& root) {
    std::pair<torch::Tensor, torch::Tensor> data = CIFAR10Test::load(root);

    images = std::move(data.first);
    targets = std::move(data.second);
}

torch::data::Example<> CIFAR10Test::get(size_t index) {
    return {images[index], targets[index]};
}

torch::optional<size_t> CIFAR10Test::size() const {
    return images.size(0);
}

const torch::Tensor& CIFAR10Test::get_images() const {
    return images;
}

const torch::Tensor& CIFAR10Test::get_targets() const {
    return targets;
}

std::pair<torch::Tensor, torch::Tensor> CIFAR10Test::load(const std::string& root) {
    std::vector<char> data_buffer;

    data_buffer.reserve(CIFAR10_CONSTANTS::BYTES_PER_BATCH_FILE);

    std::string filepath = join_paths(root, CIFAR10_CONSTANTS::TEST_LOCATION);

    std::ifstream data(filepath, std::ios::binary);

    data_buffer.insert(data_buffer.end(), std::istreambuf_iterator<char>(data), {});

    debug_print(data_buffer.size());
    debug_print(CIFAR10_CONSTANTS::TEST_SIZE);
    
    TORCH_CHECK(data_buffer.size() == CIFAR10_CONSTANTS::TEST_SIZE * CIFAR10_CONSTANTS::BYTES_PER_ROW, "Unexpected file size");

    torch::Tensor targets = torch::empty(CIFAR10_CONSTANTS::TEST_SIZE, torch::kByte);
    torch::Tensor images = torch::empty({CIFAR10_CONSTANTS::TEST_SIZE, CIFAR10_CONSTANTS::CHANNELS, CIFAR10_CONSTANTS::IMAGE_ROWS, CIFAR10_CONSTANTS::IMAGE_COLS}, torch::kByte);

    for (uint32_t i=0; i<CIFAR10_CONSTANTS::TEST_SIZE; i++) {
        uint32_t start_index = i * CIFAR10_CONSTANTS::BYTES_PER_ROW;
        targets[i] = data_buffer[start_index];

        start_index += 1;
        uint32_t end_index = start_index + CIFAR10_CONSTANTS::CHANNELS * CIFAR10_CONSTANTS::BYTES_PER_CHANNEL_PER_ROW;
        std::copy(data_buffer.begin() + start_index, 
                  data_buffer.begin() + end_index, 
                  reinterpret_cast<char*>(images[i].data_ptr()));
    }


    return {images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64)};
}