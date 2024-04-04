#pragma once

#define DEBUG

#ifdef DEBUG
#define debug_print(a) std::cout << #a << " is " << (a) << "\n";
#else
#define debug_print(a)
#endif

#include <torch/torch.h>

#include <iostream>
#include <string>

std::string join_paths(const std::string& root, const std::string& suffix);

torch::Device get_device();

