# TODOS Apr 13 - 17

* Jack
    * Setup C++ script to run the language model stuff 
        * Sync w/ Russell (NLP stuff) - need some NLP transformers models
        * Setting up pruning/quantization
    * Finish w/ the C++ exes
    * I can double check verify the pruning/quantization for ResNet (including in C++)
        * Check the bits are indeed packed
        * Pruning is indeed represented in sparse
        * Check it in C++ python
    * Run Chiyue's scripts
    * Afterwards, work w/ Chiyue on experiments
    * I can start on the report outline/draft
    * Resaerch what people have done to optimie the kernels for compute/memory
        * What optimizations people have done
        * Memory bound - increase the batch size alleviate it

* Chiyue
    * Run experiments DLProf + Resnet parameterizations (save models)
    * 


* Russell
    * 


# TODOS Apr 8 - 10

* Jack
    * Look at implmentations/relevant context for transformers 
        * Open source transformer models (huggingface) run it
        * Quantization and pruning for transformers
    * C++ hook to load transformer-relevant data
    * Make C++ executable to run the necessary pruned/quantized/base models and report accuracies for both transformers and resnet
    * How to pack the numbers according to their bits in quantization


* Chiyue
    * Figure out what stats we want to pull
        * Perf
        * Nsight compute
        * Afterwards can work on (w/ Jack or Russell) on analysis scripts using example data
    * Script to pull stats from perf, Nsight compute
    * Push code to github
    * Test out Jack's code

* Russell
    * Push resnet code to github for models
    * Debug pruning quantization
    * Test out Jack's code
    * (After sync w/ Jack offline) Transformers pruning/quantization
    * Pruning
        * Underlying weight tensors -> sparse formats (COO, CSR, parameterizable)
            * See torch.sparse

        * Sanity check that when using these in inference, they are indeed sparse

# Meeting notes

## Apr 8

* Decided to use nsight compute
    * What stats to use?
    * How to pull automatically
    * Do we need to compile w/ it for more granular or no? (think it's no?)

Apr 8

* Resnet
    * Base: 
    * Pruning: 
    * Quantization:
    * C++: 
    * Script to pull metris:

* Transformers/NLP
    * Base: 
    * Pruning: 
    * Quantization: 
    * C++: 
    * Script to pull metrics:

* Analysis scripts (in python)
    * Plotting
    * Loading data

Apr 10

* Experiments:
    * Collect data (different machines)

* Data analysis
    * Graphs
    * Correlations w/ runtime

Apr 14

* Optimizations (tuning):
    * Make some changes to models

Apr 18

* Experiments v2:
    * Collect data (different machines)
    * 

* Data analysis v2
    * Graphs
    * Correlations w/ runtime

Apr 21 
* Report

* Poster

* Present

Apr 25th