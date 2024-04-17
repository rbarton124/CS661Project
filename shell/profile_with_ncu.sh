#!/bin/bash

# Script Name: profile_with_ncu.sh

# Check if an executable was provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <executable> <output_file>"
    exit 1
fi

executable=$1

output_file=$2


# Define path to ncu
ncu_path="/usr/local/NVIDIA-Nsight-Compute/ncu"

# Define python path
python_path="/root/anaconda3/envs/torch/bin/python"



# Check if the executable exists
if [ ! -f "$executable" ]; then
    echo "Error: Executable '$executable' not found."
    exit 1
fi

# Run ncu profile
echo "Profiling $executable with ncu..."
sudo $ncu_path --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section LaunchStats --section SchedulerStats --set full --section MemoryWorkloadAnalysis_Chart -o $output_file $python_path $executable

echo "Profiling complete. Results saved to $output_file.ncu-rep"

$ncu_path --import $output_file.ncu-rep >> $output_file.txt

echo "Analyzed result saved to $output_file.txt"

LOG_FILE="$output_file.txt"

DURATION=$(grep "Duration" "$LOG_FILE" | awk '{print $3}')
CACHE_HIT=$(grep "L1/TEX Hit Rate" "$LOG_FILE" | awk '{print $5}')
SM_BUSY=$(grep "SM Busy" "$LOG_FILE" | awk '{print $4}')
MEM_BUSY=$(grep "Mem Busy" "$LOG_FILE" | awk '{print $4}')

echo "$DURATION" >> duration.txt
echo "$SM_BUSY" >> sm.txt
echo "$MEM_BUSY" >> mem.txt
echo "$CACHE_HIT" >> cache.txt