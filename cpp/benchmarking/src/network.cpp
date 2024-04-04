#include "network.h"

#include <torch/torch.h>
#include <torch/script.h>

torch::jit::script::Module load_compiled_network(const std::string& path) {
    return torch::jit::load(path);
}
