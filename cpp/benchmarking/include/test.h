#pragma once

#include "common.h"

#include <torch/torch.h>

// Testing results struct
struct TestingResults {
    double accuracy, neg_log_loss;
    std::vector<int> pred_indices;
};


// Used for testing a model w/ a specified dataloader
TestingResults test_model(auto& model, const auto& test_dataloader) {
    auto device = get_device();

    model.eval();
    torch::NoGradGuard no_grad;

    model.to(device);

    TestingResults results;

    int correct_predictions = 0;
    double tot_neg_log_loss = 0.0;
    std::vector<int> all_preds, all_targets;
    size_t total_samples = 0;

    #ifdef DEBUG
    std::cout << "Starting to test model: ";
    int idx = 0;
    #endif
    for(const auto& batch: *test_dataloader) {

        #ifdef DEBUG
        idx++;
        if (idx % 10 == 0) {
            std::cout << idx << "'th batch\n";
        }
        #endif
        
        auto data = batch.data.to(device), targets = batch.target.to(device);

        auto outputs = model.forward({data}).toTensor();

        auto preds = outputs.argmax(1);

        correct_predictions += preds.eq(targets).sum().item().toInt();

        auto loss = torch::nn::functional::cross_entropy(outputs, targets, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum));

        tot_neg_log_loss += loss.item().toDouble();


        // --------- THIS CAN BE PERF BOTTLENECK
        // auto preds_cpu = preds.to(torch::kCPU, torch::kInt64, true, true);
        // auto targets_cpu = targets.to(torch::kCPU, torch::kInt64, true, true);

        // std::vector<int64_t> preds_vec(preds_cpu.data_ptr<int64_t>(), preds_cpu.data_ptr<int64_t>() + preds_cpu.numel());
        // std::vector<int64_t> targets_vec(targets_cpu.data_ptr<int64_t>(), targets_cpu.data_ptr<int64_t>() + targets_cpu.numel());

        // all_preds.insert(all_preds.end(), preds_vec.begin(), preds_vec.end());
        // all_targets.insert(all_targets.end(), targets_vec.begin(), targets_vec.end());

        total_samples += preds.size(0);
        // END BOTTLENECK

        // all_preds.insert(all_preds.end(), preds.cpu().data_ptr(), preds.cpu().data_ptr() + preds.numel());

        // all_targets.insert(all_targets.end(), targets.cpu().data_ptr(), targets.cpu().data_ptr() + targets.numel());
    }


    double accuracy = static_cast<double>(correct_predictions) / total_samples;
    double neg_log_loss = tot_neg_log_loss / total_samples;

    results.accuracy = accuracy;
    results.neg_log_loss = neg_log_loss;
    // results.pred_indices = all_preds;

    return results;
}