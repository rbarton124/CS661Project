#pragma once

#ifdef DEBUG
#define debug_print(a) std::cout << #a << " is " << (a) << "\n";
#else
#define debug_print(a)
#endif

#include <torch/torch.h>

#include <iostream>
#include <string>

// Joins two strings into a singular string as if they were paths and a suffix
// Takes care of the "/" character in between
// Only works on linux. Likely will not work on windows
std::string join_paths(const std::string& root, const std::string& suffix);

// "cuda" if gpu else "cpu"
torch::Device get_device();

