#pragma once

#include "common.h"
#include "cifar10.h"
#include "network.h"
#include "test.h"



/* 
Method for testing loading cifar10
If the right data is present in the directory, this function should run w/ out crashing
*/
void test_load_cifar10(const std::string& path);

/*
Method for testing loading a compieled network. 

You'd run an example input tensor through it, and it will check that it has the same shape
as the example output tesnor that you give it.
*/
void test_load_compiled_network(const std::string& path, torch::Tensor example_input, torch::Tensor example_output);


/*
Method for testing the test function

Pass in a path for cifar10, and pass in the path for the .pt compiled jit model

It will test loading the cifar10 dataset, loading the model at the specified model path,
and then it will attempt to run the test function. If it doesn't crash and prints
accuracies you'd expect, it should be ok.
*/
void test_testing(const std::string& cifar10_path, const std::string& model_path);


