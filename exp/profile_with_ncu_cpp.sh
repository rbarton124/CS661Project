#!/bin/bash

# Script Name: profile_with_ncu.sh

# Check if an model was provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model> <output_file>"
    exit 1
fi

model=$1

output_file=$2


# Define path to ncu
ncu_path="ncu"




# Check if the model exists
if [ ! -f "$model" ]; then
    echo "Error: model '$model' not found."
    exit 1
fi

# Run ncu profile
echo "Profiling $model with ncu..."
$ncu_path --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section LaunchStats --section SchedulerStats --set full --section MemoryWorkloadAnalysis_Chart -o $output_file ./torch_cpp_benchmark cifar10 $model /home/cw541/661/project/CS661Project/cpp/benchmarking/data/cifar-10-batches-bin 64

echo "Profiling complete. Results saved to $output_file.ncu-rep"

$ncu_path --import $output_file.ncu-rep >> $output_file.txt

echo "Analyzed result saved to $output_file.txt"

LOG_FILE="$output_file.txt"

DURATION=$(grep "Duration" "$LOG_FILE" | awk '{print $3}')
CACHE_HIT=$(grep "L1/TEX Hit Rate" "$LOG_FILE" | awk '{print $5}')
SM_BUSY=$(grep "SM Busy" "$LOG_FILE" | awk '{print $4}')
MEM_BUSY=$(grep "Mem Busy" "$LOG_FILE" | awk '{print $4}')

echo "$DURATION" >> ${output_file}_duration.txt
echo "$SM_BUSY" >> ${output_file}_sm.txt
echo "$MEM_BUSY" >> ${output_file}_mem.txt
echo "$CACHE_HIT" >> ${output_file}_cache.txt