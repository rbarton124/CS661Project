#pragma once

#include "common.h"
#include <torch/torch.h>


// Loads a jit compiled network from a given path. For compatibility purposes please use relative paths
torch::jit::script::Module load_compiled_network(const std::string& path);