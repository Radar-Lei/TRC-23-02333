python -u main.py \
    --is_training 0 \
    --model STTN\
    \
    --root_path ./dataset/PeMS7_228 \
    --freq 5min \
    --data_shrink 1 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.3 \
    \
    --enc_in 228\
    --c_out 228\
    --d_model 32 \
    --embed learned \
    --dropout 0.1 \
    \
    --trans_channels 64 \
    --trans_layers 2 \
    --nheads 4 \
    --spa_pos_emb_dim 32 \
    \
    --batch_size 16\
    --es_patience 50\
    --lr_patience 10 \
    --learning_rate 0.001\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\

python -u main.py \
    --is_training 0 \
    --model STTN\
    \
    --root_path ./dataset/PeMS7_228 \
    --freq 5min \
    --data_shrink 1 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.6 \
    \
    --enc_in 228\
    --c_out 228\
    --d_model 32 \
    --embed learned \
    --dropout 0.1 \
    \
    --trans_channels 64 \
    --trans_layers 2 \
    --nheads 4 \
    --spa_pos_emb_dim 32 \
    \
    --batch_size 16\
    --es_patience 50\
    --lr_patience 10 \
    --learning_rate 0.001\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\

python -u main.py \
    --is_training 0 \
    --model STTN\
    \
    --root_path ./dataset/PeMS7_228 \
    --freq 5min \
    --data_shrink 1 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.75 \
    \
    --enc_in 228\
    --c_out 228\
    --d_model 32 \
    --embed learned \
    --dropout 0.1 \
    \
    --trans_channels 64 \
    --trans_layers 2 \
    --nheads 4 \
    --spa_pos_emb_dim 32 \
    \
    --batch_size 16\
    --es_patience 50\
    --lr_patience 10 \
    --learning_rate 0.001\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\

python -u main.py \
    --is_training 0 \
    --model STTN\
    \
    --root_path ./dataset/PeMS7_1026 \
    --freq 5min \
    --data_shrink 9 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.3 \
    \
    --enc_in 1026\
    --c_out 1026\
    --d_model 32 \
    --embed learned \
    --dropout 0.1 \
    \
    --trans_channels 64 \
    --trans_layers 2 \
    --nheads 4 \
    --spa_pos_emb_dim 32 \
    \
    --batch_size 4\
    --es_patience 50\
    --lr_patience 10 \
    --learning_rate 0.001\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\

python -u main.py \
    --is_training 0 \
    --model STTN\
    \
    --root_path ./dataset/PeMS7_1026 \
    --freq 5min \
    --data_shrink 9 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.6 \
    \
    --enc_in 1026\
    --c_out 1026\
    --d_model 32 \
    --embed learned \
    --dropout 0.1 \
    \
    --trans_channels 64 \
    --trans_layers 2 \
    --nheads 4 \
    --spa_pos_emb_dim 32 \
    \
    --batch_size 4\
    --es_patience 50\
    --lr_patience 10 \
    --learning_rate 0.001\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\

python -u main.py \
    --is_training 0 \
    --model STTN\
    \
    --root_path ./dataset/PeMS7_1026 \
    --freq 5min \
    --data_shrink 9 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.75 \
    \
    --enc_in 1026\
    --c_out 1026\
    --d_model 32 \
    --embed learned \
    --dropout 0.1 \
    \
    --trans_channels 64 \
    --trans_layers 2 \
    --nheads 4 \
    --spa_pos_emb_dim 32 \
    \
    --batch_size 4\
    --es_patience 50\
    --lr_patience 10 \
    --learning_rate 0.001\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\


#### Seattle ####

python -u main.py \
    --is_training 0 \
    --model STTN\
    \
    --root_path ./dataset/Seattle \
    --freq 5min \
    --data_shrink 1 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.3 \
    \
    --enc_in 323\
    --c_out 323\
    --d_model 32 \
    --embed learned \
    --dropout 0.1 \
    \
    --trans_channels 64 \
    --trans_layers 2 \
    --nheads 4 \
    --spa_pos_emb_dim 32 \
    \
    --batch_size 16\
    --es_patience 50\
    --lr_patience 10 \
    --learning_rate 0.001\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\

python -u main.py \
    --is_training 0 \
    --model STTN\
    \
    --root_path ./dataset/Seattle \
    --freq 5min \
    --data_shrink 1 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.6 \
    \
    --enc_in 323\
    --c_out 323\
    --d_model 32 \
    --embed learned \
    --dropout 0.1 \
    \
    --trans_channels 64 \
    --trans_layers 2 \
    --nheads 4 \
    --spa_pos_emb_dim 32 \
    \
    --batch_size 16\
    --es_patience 50\
    --lr_patience 10 \
    --learning_rate 0.001\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\

python -u main.py \
    --is_training 0 \
    --model STTN\
    \
    --root_path ./dataset/Seattle \
    --freq 5min \
    --data_shrink 1 \
    \
    --seq_len 18 \
    --missing_pattern rcm \
    --missing_rate 0.75 \
    \
    --enc_in 323\
    --c_out 323\
    --d_model 32 \
    --embed learned \
    --dropout 0.1 \
    \
    --trans_channels 64 \
    --trans_layers 2 \
    --nheads 4 \
    --spa_pos_emb_dim 32 \
    \
    --batch_size 16\
    --es_patience 50\
    --lr_patience 10 \
    --learning_rate 0.001\
    --train_epochs 600\
    \
    --gpu 0 \
    \
    --epoch_to_vis 5\