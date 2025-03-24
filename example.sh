# REMEMBER: your df must have a "prompts" column, else, please modify the original code's column name.

export CUDA_VISIBLE_DEVICES=1,2

python cot_vllm.py \
    --model_name "FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview" \
    --tok_name "FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview" \
    --template_path "/template/extract_kor.txt" \
    --df_path "your_df_path" \
    --SAVE_DIR "your_save_path" \
    --tensor_parallel_size 2 \
    --max_model_len 32768 \
    --max_new_token 16384 \
    --batch_size 2000 \
    --num_k 10 \
    --STEERING_TOKEN "" \
    --EXTRACT_TOKEN ""
