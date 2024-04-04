#pragma once

#include "common.h"
#include <torch/torch.h>

torch::jit::script::Module load_compiled_network(const std::string& path);