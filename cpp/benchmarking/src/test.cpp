#include "test.h"

#include <vector>

// TestingResults test_model(auto& model, const auto& test_dataloader) {
//     auto device = get_device();

//     model->eval();
//     torch::NoGradGuard no_grad;

//     model->to(device);

//     TestingResults results;

//     int correct_predictions = 0;
//     double tot_neg_log_loss = 0.0;
//     std::vector<int> all_preds, all_targets;

//     #ifdef DEBUG
//     std::cout << "Starting to test model: ";
//     int idx = 0;
//     #endif
//     for(const auto& batch: *test_dataloader) {

//         #ifdef DEBUG
//         idx++;
//         if (idx % 10 == 0) {
//             std::cout << idx << "'th batch\n";
//         }
//         #endif
        
//         auto data = batch.data.to(device), targets = batch.target.to(device);

//         auto outputs = model->forward(data);

//         auto preds = outputs.argmax(1);

//         correct_predictions += preds.eq(targets).sum().item();

//         auto loss = torch::nn::functional::cross_entropy(outputs, targets, torch::kSum);

//         tot_neg_log_loss += loss.item();

//         all_preds.insert(all_preds.end(), preds.cpu().data_ptr(), preds.cpu().data_ptr() + preds.numel());

//         all_targets.insert(all_targets.end(), targets.cpu().data_ptr(), targets.cpu().data_ptr() + targets.numel());
//     }

//     size_t total_samples = test_dataloader->size().value();

//     double accuracy = static_cast<double>(correct_predictions) / total_samples;
//     double neg_log_loss = tot_neg_log_loss / total_samples;

//     results.accuracy = accuracy;
//     results.neg_log_loss = neg_log_loss;
//     results.pred_indices = all_preds;

//     return results;
// }

void display_results(const TestingResults& results) {
    std::cout << "Accuracy: " << results.accuracy << '\n';
}