#pragma once

#include "common.h"
#include "cifar10.h"
#include "network.h"
#include "test.h"


void test_load_cifar10(const std::string& path);

void test_load_compiled_network(const std::string& path, torch::Tensor example_input, torch::Tensor example_output);

void test_testing(const std::string& cifar10_path, const std::string& model_path);


