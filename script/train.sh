# first train bart chinese
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
    --overwrite_output_dir \
    --save_strategy epoch \
    --num_train_epochs 10\
    --output_dir checkpoint/google_mt5_small_3e_4_10\
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --evaluation_strategy epoch \
    # --report_to wandb \
    # --run_name google_mt5_small_3e_4_10 \
    # --logging_strategy epoch \