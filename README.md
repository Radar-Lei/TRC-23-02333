# TRC-23-02333
# TRC-23-02333

This repository contains the prototype codes for our submission "TRC-23-023333" to the journal *Transportation Research Part C*. Currently, we present only the core components of our implementation, including the estimation processes using our proposed model, two baseline models, and an ablation study. Once our paper is accepted, we will upload the complete implementations including experiments comparing all baselines. You can test the currently available experiments using the following commands on a Linux-based system:

```bash
bash ./scripts/PeMS7_228.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_228_log.txt
bash ./scripts/PeMS7_1026.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_1026_log.txt
bash ./scripts/Seattle.sh > $(date +'%y%m%d-%H%M%S')_Seattle_log.txt

bash ./scripts/Linear_interpolation.sh > $(date +'%y%m%d-%H%M%S')_Linear_interpolation_log.txt

# For ablation analysis
bash ./scripts/PeMS7_228_wo_tem.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_228_wo_tem_log.txt
bash ./scripts/PeMS7_228_wo_spa.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_228_wo_spa_log.txt
```

Please do not share our code elsewhere; this implementation is currently only for reviewers to check.

## Multi-GPUs Implementation

We provide an implementation that leverages multiple GPUs for parallel accelerated training and sample generation in another branch named "multi_GPU". You can switch to that branch if you're equipped with multiple GPUs. It is important to note that to ensure the visual order of the time series of the samples generated during validation and testing is maintained, we only use two GPUs in these phases.

## Data and Training Logs

We share the data we used and the training logs via Google Drive. The link includes our trained models and intermediate results produced during training. By visualizing these intermediate results, you can observe that the performance improves progressively throughout the training process.

```bash
bash ./scripts/PeMS7_228.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_228_log.txt
bash ./scripts/PeMS7_1026.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_1026_log.txt
bash ./scripts/Seattle.sh > $(date +'%y%m%d-%H%M%S')_Seattle_log.txt

bash ./scripts/Linear_interpolation.sh > $(date +'%y%m%d-%H%M%S')_Linear_interpolation_log.txt

# for ablation analysis
bash ./scripts/PeMS7_228_wo_tem.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_228_wo_tem_log.txt
bash ./scripts/PeMS7_228_wo_spa.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_228_wo_spa_log.txt
```

**Please do not share our code elsewhere; this implementation is currently only for reviewers to check.**

## Multi-GPUs Implementation
We provide an implementation that leverages multiple GPUs for parallel accelerated training and sample generation in another branch named "multi_GPU". You can switch to that branch if you're equipped with multiple GPUs. It is important to note that to ensure the visual order of the time series of the samples generated during validation and testing is maintained, we only use two GPUs in these phases.

## Data and Training Logs
We share the data we used and the training logs via Google Drive. The link includes our trained models and intermediate results produced during training. By visualizing these intermediate results, you can observe that the performance improves progressively throughout the training process.

![Validation results in early epochs](figs/early_epoch.png)
<p align="center">Validation results in early epochs</p>