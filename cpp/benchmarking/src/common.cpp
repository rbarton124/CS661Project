#include "common.h"

#include <string>

std::string join_paths(const std::string& root, const std::string& suffix) {
    std::string result = root;
    if (root.back() != '/')
        result.push_back('/');
    
    result += suffix;
    return result; 
}

torch::Device get_device() {
    if (torch::cuda::is_available())
        return torch::Device(torch::kCUDA);
    else 
        return torch::Device(torch::kCPU);
}

