import argparse
import torch
from imputation import Exp_Imputation
import datetime
import json
import os
"""
bash ./scripts/PeMS7_228.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_228_log.txt
bash ./scripts/PeMS7_1026.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_1026_log.txt
bash ./scripts/Seattle.sh > $(date +'%y%m%d-%H%M%S')_Seattle_log.txt

bash ./scripts/Linear_interpolation.sh > $(date +'%y%m%d-%H%M%S')_Linear_interpolation_log.txt

bash ./scripts/STTN.sh > $(date +'%y%m%d-%H%M%S')_STTN_log.txt
"""

fix_seed = 42

parser = argparse.ArgumentParser(description='Traffic State Estimation at Sensor-free Locations')

# basic config
parser.add_argument('--task_name', type=str, default='imputation',
                    help='task name, options:[prediction, imputation]')
parser.add_argument('--is_training', type=int, default=0, help='status, options:[0:training, 1:testing, 2:pred]')
parser.add_argument('--model', type=str, default='CDSTE',
                        help='model name, options: [CDSTE, CDSTE_wo_tem, CDSTE_wo_spa, CSDI, STTN, TimeGrad, DeepAR]')

# data loader
parser.add_argument(
    '--root_path', type=str, default='./dataset/PeMS7_228', help='root path of the data file'
    ) # ./dataset/PeMS7_228 ./dataset/PeMS7_1026 ./dataset/Seattle
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--freq', type=str, default='5min',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--data_shrink', type=int, default=6, help='reduce the numbder of samples')

# imputation task
parser.add_argument('--seq_len', type=int, default=18, help='input sequence length')
# scm, structurally column missing, might better fit real-world scenarios
parser.add_argument('--missing_pattern', type=str, default='rcm', 
                    help='missing pattern, options:[rcm:randomly column missing, scm:structurally column missing]')
parser.add_argument('--missing_rate', type=float, default=0.3, help='missing rate')
parser.add_argument('--fixed_seed', type=int, default=fix_seed)

# diffusion
parser.add_argument('--diff_schedule', type=str, default='quad', help='schedule for diffusion, options:[quad, linear]')
parser.add_argument('--diff_steps', type=int, default=100, help='num of diffusion steps')
parser.add_argument('--diff_samples', type=int, default=32, help='num of generated samples')
parser.add_argument('--beta_start', type=float, default=0.0001, help='start beta for diffusion, 0.0001')
parser.add_argument('--beta_end', type=float, default=0.2, help='end beta for diffusion, 0.1, 0.2, 0.3, 0.4')
parser.add_argument('--sampling_shrink_interval', type=int, default=4, help='shrink interval for sampling')

# model define
parser.add_argument('--locations', type=int, default=228, help='number of locations')
parser.add_argument('--d_model', type=int, default=32, help='dimension of TimesNet module') # 
parser.add_argument('--d_ff', type=int, default=16, help='dimension of fcn') # FC network, 
parser.add_argument('--top_k', type=int, default=2, help='for TimesBlock') # 5
parser.add_argument('--num_kernels', type=int, default=2, help='for Inception') # 6
parser.add_argument('--embed', type=str, default='learned',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

parser.add_argument('--channels', type=int, default=64, help='channels of Transformer') # 
parser.add_argument('--layers', type=int, default=2, help='layers of Transformer blocks') #
parser.add_argument('--nheads', type=int, default=4, help='Numher of Multi-head') #
parser.add_argument('--diff_emb_dim', type=int, default=32, help='Embbedding dim of the diffusion step') #
parser.add_argument('--spa_pos_emb_dim', type=int, default=32, help='Embedding dim for features (different locations)') # attention computation along feature axis

# optimization
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--itr', type=int, default=1, help='experiments times') # num of experiments
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--es_patience', type=int, default=32, help='early stopping patience')
parser.add_argument('--lr_patience', type=int, default=50, help='learning rate decreasing patience')
parser.add_argument('--learning_rate', type=float, default=0.002, help='optimizer learning rate')
parser.add_argument('--train_epochs', type=int, default=600, help='train epochs')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

# visual
parser.add_argument('--epoch_to_vis', type=int, default=5) # epoch invertal to evaluate and visualize

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

if args.task_name == 'imputation':
    Exp = Exp_Imputation

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if args.is_training == 0:
    # train
    for ii in range(args.itr):
        setting = '{}_{}_{}_mr{}'.format(
            current_time,
            args.root_path.split('/')[-1],
            args.model,
            args.missing_rate,
            )
        
        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        
        print(json.dumps(vars(args), indent=4))
        
        folder_path = './vali_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        with open(folder_path + "model_config.json", "w") as f:
            f.write(json.dumps(vars(args), indent=4))
        
        exp.train(setting)

        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting)
        torch.cuda.empty_cache() 

elif args.is_training == 2:
    # pred
    ii = 0
    setting = ''

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.pred(setting)
    torch.cuda.empty_cache()