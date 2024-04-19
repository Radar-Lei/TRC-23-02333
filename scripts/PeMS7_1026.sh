# beta_start and beta_end and diff_steps are very important for the performance of generation
# if you found the values from the decoder is exploding, you may need to reduce beta_start and beta_end, or reduce diff_steps, 
# to eventually reduce the variance of the sampling distribution (diffusion rate)
python -u main.py \
    --is_training 0 \
    --model DiffusionBase\
    \
    --root_path ./dataset/PeMS7_1026 \
    --freq t \
    --data_shrink 9 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.3 \
    \
    --diff_schedule quad \
    --diff_steps 100 \
    --diff_samples 32 \
    --beta_start 0.0001 \
    --beta_end 0.2 \
    --sampling_shrink_interval 8 \
    \
    --e_layers 1\
    --enc_in 1026\
    --c_out 1026\
    --d_model 32 \
    --d_ff 32 \
    --top_k 5 \
    --num_kernels 6 \
    --embed timeF \
    --dropout 0.1 \
    \
    --trans_channels 64 \
    --trans_layers 2 \
    --nheads 4 \
    --diff_emb_dim 32 \
    --spa_pos_emb_dim 32 \
    \
    --batch_size 4\
    --patience 30 \
    --learning_rate 0.001\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\