export CUDA_VISIBLE_DEVICES=0 && python code/run_summarization.py \
    --seed 42 \
    --model_name_or_path google/mt5-small \
    --do_train \
    --do_eval \
    --text_column maintext \
    --summary_column title \
    --learning_rate 3e-4 \
    --train_file data/train.jsonl \
    --train_test_split 0.05 \
    --max_source_length 512 \
    --max_target_length 64 \
    --source_prefix "summarize: " \
    --evaluation_strategy epoch \
    --num_train_epochs 10\
    --output_dir checkpoint/google_mt5_small_3e_4_10\
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --save_strategy epoch \
    --predict_with_generate \
    --report_to wandb \
    --run_name google_mt5_small_3e_4_10 \
    --logging_strategy epoch \


# export CUDA_VISIBLE_DEVICES=1 && python code/run_summarization.py \
#     --seed 42 \
#     --model_name_or_path google/mt5-base \
#     --do_train \
#     --do_eval \
#     --text_column maintext \
#     --summary_column title \
#     --learning_rate 5e-5 \
#     --train_file data/train.jsonl \
#     --train_test_split 0.05 \
#     --max_source_length 384 \
#     --max_target_length 64 \
#     --source_prefix "summarize: " \
#     --evaluation_strategy epoch \
#     --num_train_epochs 6\
#     --output_dir checkpoint/google_mt5_base_5e-5\
#     --per_device_train_batch_size=4 \
#     --per_device_eval_batch_size=4 \
#     --overwrite_output_dir \
#     --save_strategy epoch \
#     --predict_with_generate \
#     --report_to wandb \
#     --run_name google_mt5_base_5e-5 \
#     --logging_strategy epoch \

# first train bart chinese
# export CUDA_VISIBLE_DEVICES=1 && python code/run_summarization.py \
#     --seed 42 \
#     --model_name_or_path fnlp/bart-large-chinese \
#     --do_train \
#     --do_eval \
#     --text_column maintext \
#     --summary_column title \
#     --learning_rate 2e-5 \
#     --train_file data/train.jsonl \
#     --train_test_split 0.05 \
#     --max_source_length 512 \
#     --max_target_length 64 \
#     --evaluation_strategy epoch \
#     --num_train_epochs 10\
#     --output_dir checkpoint/fnlp_bart-large-chinese\
#     --per_device_train_batch_size=8 \
#     --per_device_eval_batch_size=4 \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --report_to wandb \
#     --run_name fnlp_bart-large-chinese \
#     --logging_strategy epoch \


# train facebook mbart
# export CUDA_VISIBLE_DEVICES=1 && python code/run_summarization.py \
#     --seed 42 \
#     --model_name_or_path facebook/mbart-large-cc25 \
#     --lang zh_CN \
#     --do_train \
#     --do_eval \
#     --text_column maintext \
#     --summary_column title \
#     --learning_rate 3e-5 \
#     --train_file data/train.jsonl \
#     --train_test_split 0.05 \
#     --max_source_length 512 \
#     --max_target_length 64 \
#     --evaluation_strategy epoch \
#     --num_train_epochs 5\
#     --output_dir checkpoint/facebook_mbart-large-cc25\
#     --per_device_train_batch_size=8 \
#     --per_device_eval_batch_size=4 \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --report_to wandb \
#     --run_name facebook_mbart-large-cc25 \
#     --logging_strategy epoch \