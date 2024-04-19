# beta_start and beta_end and diff_steps are very important for the performance of generation
# if you found the values from the decoder is exploding, you may need to reduce beta_start and beta_end, or reduce diff_steps, 
# to eventually reduce the variance of the sampling distribution (diffusion rate)
python -u main.py \
    --is_training 0 \
    --model CDSTE\
    \
    --root_path ./dataset/PeMS7_228 \
    --freq 5min \
    --data_shrink 3 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.3 \
    \
    --diff_schedule quad \
    --diff_steps 100 \
    --diff_samples 50 \
    --beta_start 0.0001 \
    --beta_end 0.2 \
    --sampling_shrink_interval 4 \
    \
    --locations 228\
    --d_model 32 \
    --d_ff 8 \
    --top_k 2 \
    --num_kernels 2 \
    --embed learned \
    --dropout 0.1 \
    \
    --channels 64 \
    --layers 2 \
    --nheads 4 \
    --diff_emb_dim 32 \
    --spa_pos_emb_dim 64 \
    \
    --batch_size 16\
    --es_patience 100\
    --lr_patience 30\
    --learning_rate 0.003\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\

python -u main.py \
    --is_training 0 \
    --model CDSTE\
    \
    --root_path ./dataset/PeMS7_228 \
    --freq 5min \
    --data_shrink 3 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.6 \
    \
    --diff_schedule quad \
    --diff_steps 100 \
    --diff_samples 50 \
    --beta_start 0.0001 \
    --beta_end 0.2 \
    --sampling_shrink_interval 4 \
    \
    --locations 228\
    --d_model 32 \
    --d_ff 8 \
    --top_k 2 \
    --num_kernels 2 \
    --embed learned \
    --dropout 0.1 \
    \
    --channels 64 \
    --layers 2 \
    --nheads 4 \
    --diff_emb_dim 32 \
    --spa_pos_emb_dim 64 \
    \
    --batch_size 16\
    --es_patience 100\
    --lr_patience 30\
    --learning_rate 0.003\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\

python -u main.py \
    --is_training 0 \
    --model CDSTE\
    \
    --root_path ./dataset/PeMS7_228 \
    --freq 5min \
    --data_shrink 3 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.75 \
    \
    --diff_schedule quad \
    --diff_steps 100 \
    --diff_samples 50 \
    --beta_start 0.0001 \
    --beta_end 0.2 \
    --sampling_shrink_interval 4 \
    \
    --locations 228\
    --d_model 32 \
    --d_ff 8 \
    --top_k 2 \
    --num_kernels 2 \
    --embed learned \
    --dropout 0.1 \
    \
    --channels 64 \
    --layers 2 \
    --nheads 4 \
    --diff_emb_dim 32 \
    --spa_pos_emb_dim 64 \
    \
    --batch_size 16\
    --es_patience 100\
    --lr_patience 30\
    --learning_rate 0.003\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\