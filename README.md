# TRC-23-02333

This repository contains the prototype codes for our submission "TRC-23-023333" to the journal *Transportation Research Part C*. Currently, we present only the core components of our implementation, including the estimation processes using our proposed model, two baseline models, and an ablation study. 

## Project Overview

This project focuses on traffic state estimation at sensor-free locations using advanced deep learning techniques. The implementation includes:

- A novel CDSTE (Conditional Diffusion-based Spatio-Temporal Estimation) model
- Baseline models for comparison
- Ablation studies to analyze model components
- Multi-GPU support for efficient training
- Comprehensive evaluation metrics and visualization tools

## Requirements

- Python 3.8+
- PyTorch 2.0.1
- CUDA 11.8
- Linux-based operating system
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/TRC-23-02333.git
cd TRC-23-02333
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can test the currently available experiments using the following commands:

### Main Experiments
```bash
# PeMS7-228 dataset
bash ./scripts/PeMS7_228.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_228_log.txt

# PeMS7-1026 dataset
bash ./scripts/PeMS7_1026.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_1026_log.txt

# Seattle dataset
bash ./scripts/Seattle.sh > $(date +'%y%m%d-%H%M%S')_Seattle_log.txt

# Linear interpolation baseline
bash ./scripts/Linear_interpolation.sh > $(date +'%y%m%d-%H%M%S')_Linear_interpolation_log.txt
```

### Ablation Studies
```bash
# Without temporal component
bash ./scripts/PeMS7_228_wo_tem.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_228_wo_tem_log.txt

# Without spatial component
bash ./scripts/PeMS7_228_wo_spa.sh > $(date +'%y%m%d-%H%M%S')_PeMS7_228_wo_spa_log.txt
```

## Multi-GPUs Implementation

We provide an implementation that leverages multiple GPUs for parallel accelerated training and sample generation in another branch named "multi_GPU". You can switch to that branch if you're equipped with multiple GPUs. It is important to note that to ensure the visual order of the time series of the samples generated during validation and testing is maintained, we only use two GPUs in these phases.

To use the multi-GPU version:
```bash
git checkout multi_GPU
```

## Data and Training Logs

We share the data we used and the training logs via [Google Drive](https://drive.google.com/drive/folders/14VPjNlQQRd5FCXXHrBPYbety9fiWx7--?usp=drive_link). The link includes:
- Preprocessed datasets
- Trained model checkpoints
- Training logs
- Intermediate results
- Visualization outputs

By visualizing these intermediate results, you can observe that the performance improves progressively throughout the training process.

## Model Architecture

The CDSTE model consists of several key components:
1. Temporal Encoder: Captures temporal dependencies in traffic data
2. Spatial Encoder: Models spatial relationships between locations
3. Diffusion Module: Generates high-quality imputations
4. Attention Mechanism: Weights different features and time steps

## Evaluation Metrics

The model is evaluated using multiple metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- CRPS (Continuous Ranked Probability Score)

## Results Visualization

The training process can be monitored through visualizations:

![Validation results in early epochs](figs/early_epoch.png)
<p align="center">Validation results in early epochs</p>

![Validation results in late epochs](figs/late_epoch.png)
<p align="center">Validation results in late epochs</p>

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@article{lei2024conditional,
  title={A conditional diffusion model for probabilistic estimation of traffic states at sensor-free locations},
  author={Lei, Da and Xu, Min and Wang, Shuaian},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={166},
  pages={104798},
  year={2024},
  publisher={Elsevier}
}
```

## License

This project is currently under review. Please do not share our code elsewhere; this implementation is currently only for reviewers to check.

## Contact

For any questions or issues, please contact [greatradar@gmail.com](mailto:greatradar@gmail.com).

## Acknowledgments

We would like to thank the reviewers for their valuable feedback and suggestions.