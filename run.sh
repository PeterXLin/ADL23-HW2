# bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
test_path=$1
output_path=$2

python ./code/predict.py \
    --test_path $test_path \
    --output_path $output_path \
    --model_path "checkpoint/google_mt5_small_3e_4_10/checkpoint-12890" \
    --decoding_strategy "beam_search" \
    --batch_size 16 \
    --max_source_length 384 \
    --max_target_length 64 \