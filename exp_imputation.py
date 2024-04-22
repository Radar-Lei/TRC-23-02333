import torch
from exp_base import Exp_Basic
import torch.distributed
from torch.distributed import barrier
import os
import torch.nn as nn
from dataloader import data_provider
from torch import optim
import numpy as np
from utils import metric, EarlyStopping, plot_subplots, daily_plot_subplots
import time

import torch.distributed as dist
from torch.cuda.amp import GradScaler

"""
if the saved model is saving with 'model.state_dict()', instead of 'model.module.state_dict()'.
It was wrapped with DDP, we need to use OrderedDict to remove the 'module.' in the key.
"""


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        """
        super is used to call  you're calling the __init__ method of
        the parent class (which is Exp_Basic) of Exp_Imputation
        """
        super(Exp_Imputation, self).__init__(args)
        if self.args.root_path in [
            "./dataset/PeMS7_228",
            "./dataset/PeMS7_1026",
            "./dataset/Seattle",
        ]:
            self.L_d = 288

    def _build_model(self):
        """
        set_device: set the default device to the current local rank, then we could assign any
        tensor to the device without specifying the device. e.g.,  [any tensor].cuda()
        """
        if self.args.use_multi_gpu and self.args.use_gpu:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            model = self.model_dict[self.args.model].Model(self.args).cuda()

            if len(self.args.trained_model) > 1:
                print("loading model")
                path = os.path.join(self.args.checkpoints, self.args.trained_model)
                # when trying to load the model and continue to train on Multi-GPU, must add map_location
                checkpoint = torch.load(
                    os.path.join(path, "checkpoint.pth"),
                    map_location=f"cuda:{self.local_rank}",
                )
                model.load_state_dict(checkpoint["model_state_dict"])

            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # batch_norm sync
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank]
            )
        self.scaler = GradScaler()  # mixed precision training

        if self.local_rank == 0:
            num_params = sum(p.numel() for p in model.parameters())
            print("Number of parameters in current model:", num_params)
            
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # Adadelta, Adagrad, RMSprop, Adam, AdamW, NADAM, RAdam, SGD
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6
        )

        if self.args.trained_model != "":
            print("loading optimizer")
            path = os.path.join(self.args.checkpoints, self.args.trained_model)
            # must add map_location for continue training on Multi-GPU by loading the trained model
            checkpoint = torch.load(
                os.path.join(path, "checkpoint.pth"),
                map_location=f"cuda:{self.local_rank}",
            )
            model_optim.load_state_dict(checkpoint["optimizer_state_dict"])
        return model_optim

    def _get_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.9, patience=self.args.lr_patience, verbose=True
        )
        if self.args.trained_model != "":
            print("loading scheduler")
            path = os.path.join(self.args.checkpoints, self.args.trained_model)
            checkpoint = torch.load(
                os.path.join(path, "checkpoint.pth"),
                map_location=f"cuda:{self.local_rank}",
            )
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return scheduler

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _sm_mask_generator(self, actual_mask, reserve_indices, missing_rate):
        """
        generate the missing mask for SM missing pattern,
        should follow the same strategy as dataloader to
        select cols to be structurally missing, but without
        fixed random seed for training set diversity.

        return: (B,L,K) as the cond_mask in model training
        """
        # actual_mask: (B,L,K)
        copy_mask = actual_mask.clone()
        _, _, dim_K = copy_mask.shape
        available_features = [i for i in range(dim_K) if i not in reserve_indices]
        # every time randomly
        np.random.seed(None)
        selected_features = np.random.choice(
            available_features,
            round(len(available_features) * missing_rate),
            replace=False,
        )
        copy_mask[:, :, selected_features] = 0

        return copy_mask, selected_features

    def _quantile_loss(self, target, forecast, q: float, eval_points) -> float:
        return 2 * np.sum(
            np.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
        )

    def _calc_denominator(self, target, eval_points):
        return np.sum(np.abs(target * eval_points))

    def _calc_quantile_CRPS(self, target, forecast, eval_points):
        quantiles = np.arange(0.05, 1.0, 0.05)
        denom = self._calc_denominator(target, eval_points)
        CRPS = 0
        for i in range(len(quantiles)):
            q_pred = []
            for j in range(len(forecast)):
                q_pred.append(np.quantile(forecast[j : j + 1], quantiles[i], axis=1))
            q_pred = np.concatenate(q_pred, axis=0)
            q_loss = self._quantile_loss(target, q_pred, quantiles[i], eval_points)
            CRPS += q_loss / denom
        return CRPS / len(quantiles)

    def _get_quantile(self, samples, q, axis=1):
        return np.quantile(samples, q, axis=axis)

    def _quantile(self, samples, all_target_np, all_given_np):
        qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
        quantiles_imp = []
        for q in qlist:
            quantiles_imp.append(
                self._get_quantile(samples, q, axis=1) * (1 - all_given_np)
                + all_target_np * all_given_np
            )
        return quantiles_imp

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        if self.local_rank == 1:
            vali_data, vali_loader = self._get_data(flag="val")
        if self.local_rank == 0:
            test_data, test_loader = self._get_data(flag="test")

        if self.args.missing_pattern == "rcm":
            _, K = train_data[0][0].shape
            np.random.seed(self.args.fixed_seed)
            reserve_indices = np.random.choice(
                range(K), round(K * self.args.missing_rate), replace=False
                )
            if self.local_rank == 0:
                print(
                    "Num of reserved locations for model eval: {} out of {} locations \nwith seed {}".format(
                        len(reserve_indices), K, self.args.fixed_seed
                    )
                )
                print("Reserved location ids:", reserve_indices)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and self.local_rank == 0:
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.es_patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = self._get_scheduler(model_optim)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = torch.tensor(0.0).cuda()
            # num of batches
            train_size = torch.tensor(len(train_loader)).float().cuda()

            # train_loader.sampler.set_epoch(epoch)  # prevent sampling bug

            self.model.train()
            epoch_time = time.time()
            # batch_x: time series itself, batch_x_mark: time stamps
            for i, (batch_x, batch_x_mark, actual_mask, weight_A) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().cuda()
                batch_x_mark = batch_x_mark.float().cuda()
                actual_mask = actual_mask.float().cuda()

                if self.args.missing_pattern == "rcm":
                    # randomly column missing
                    actual_mask[:, :, reserve_indices] = 0
                    mask, selected_cols = self._sm_mask_generator(
                        actual_mask, reserve_indices, self.args.missing_rate
                    )

                mask = mask.float().cuda()
                mask = actual_mask * mask
                target_mask = actual_mask - mask
                weight_A = weight_A.float().cuda()

                # remember that in the forward process, we compute the loss between the predicted noise nad the actual noise
                outputs, curr_noise = self.model(
                    batch_x, batch_x_mark, mask, target_mask, weight_A
                )

                loss = criterion(
                    outputs[:,:,selected_cols], curr_noise[:,:,selected_cols]
                )
                train_loss += loss

                loss.backward()
                model_optim.step()

            # end time for the current epoch
            curr_epoch_time = time.time()
            if self.local_rank == 0:
                print(
                    "Epoch: {} training cost time: {:.2f}".format(
                        epoch + 1, curr_epoch_time - epoch_time
                    )
                )

            dist.reduce(train_loss, 0, op=dist.ReduceOp.SUM)  # sum loss from all gpus
            dist.reduce(train_size, 0, op=dist.ReduceOp.SUM)

            if ((epoch + 1) % self.args.epoch_to_vis == 0) or (epoch <= 4):
                # epoch
                if self.local_rank == 1:
                    vali_metrics = self.vali(
                        vali_data, vali_loader, reserve_indices, epoch + 1, setting
                    )

                if self.local_rank == 0:
                    test_metrics = self.vali(
                        test_data, test_loader, reserve_indices, epoch + 1
                    )             
                
                barrier()
                
                if self.local_rank == 1:

                    print(
                        "Epoch: {0}, eval cost time: {1:.2f} | Train Loss: {2:.2f}"
                        .format(
                        epoch + 1, time.time()-curr_epoch_time, train_loss / train_size
                        )
                    )
                    
                    print(
                        "| Vali MAE (mean): {0:.2f} RMES (mean): {1:.2f} MAPE (mean): {2:.2f}".format(
                            vali_metrics[0], vali_metrics[1], vali_metrics[2]
                        )
                    )
                    print(
                        "| Vali MAE (median): {0:.2f} RMES (median): {1:.2f} MAPE (median): {2:.2f} \n| CRPS (Vali): {3:.2f}".format(
                            vali_metrics[3], vali_metrics[4], vali_metrics[5], vali_metrics[6]
                        )
                    )
                    
                barrier()
                
                if self.local_rank == 0:                    
                    print(
                        "| Test MAE (mean): {0:.2f} RMES (mean): {1:.2f} MAPE (mean): {2:.2f}".format(
                            test_metrics[0], test_metrics[1], test_metrics[2]
                        )
                    )
                    print(
                        "| Test MAE (median): {0:.2f} RMES (median): {1:.2f} MAPE (median): {2:.2f} \n| CRPS (Test): {3:.2f}".format(
                            test_metrics[3], test_metrics[4], test_metrics[5], test_metrics[6]
                        )
                    )
                    
                barrier()
                
                if self.local_rank == 1:
                    early_stopping(
                        vali_metrics[6], self.model, model_optim, scheduler, path
                    )

                    if early_stopping.early_stop:
                        print("Early stopping")
                        print("-----------------------------------------")
                        print("")
                        break
                    
            else:
                if self.local_rank == 1:
                    print(
                        "Epoch: {0}, Steps: {1} | Train Loss: {2:.4f}".format(
                            epoch + 1, train_steps, train_loss / train_size
                        )
                    )

            scheduler.step(train_loss)

        dist.destroy_process_group()

        return self.model

    def vali(self, vali_data, vali_loader, reserve_indices, epoch, setting=None):
        all_outputs, all_targets, all_medians, all_means, all_masks, all_obs_masks = [], [], [], [], [], []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_x_mark, actual_mask, weight_A) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().cuda()
                batch_x_mark = batch_x_mark.float().cuda()

                # random mask
                B, L, K = batch_x.shape

                if self.args.missing_pattern == "rcm":
                    # randomly structurally missing
                    mask = actual_mask.clone()
                    mask[:, :, reserve_indices] = 0

                mask = mask.float().cuda()
                # mask for the NaN values in the original data
                actual_mask = actual_mask.float().cuda()
                mask = actual_mask * mask
                target_mask = actual_mask - mask
                weight_A = weight_A.float().cuda()

                # outputs is of shape (B, n_samples, L_hist, K)
                # whether to use the sampling shrink interval to accelerate the sampling process
                """
                You should call the evaluate_acc method on the underlying model, not the DistributedDataParallel wrapper.
                This is because the evaluate_acc method is defined in the Model class in models/DiffusionBase.py, 
                but it's not accessible when the model is wrapped with DistributedDataParallel.
                """
                if self.args.sampling_shrink_interval > 1:
                    outputs = self.model.module.evaluate_acc(
                        batch_x, batch_x_mark, mask, target_mask, weight_A
                    )
                else:
                    outputs = self.model.module.evaluate(
                        batch_x, batch_x_mark, mask, target_mask, weight_A
                    )

                # eval
                B, n_samples, L, K = outputs.shape
                # unnormalize outputing samples and current target
                outputs = vali_data.inverse_transform(
                    outputs.detach().cpu().numpy().reshape(B * n_samples * L, K)
                ).reshape(B, n_samples, L, K)

                # current target of shape (B, L_hist, K)
                c_target = vali_data.inverse_transform(
                    batch_x.detach().cpu().numpy().reshape(B * L, K)
                ).reshape(B, L, K)

                # (B, n_samples, L_hist, K) -> (B, L_hist, K)
                samples_median = np.median(outputs, axis=1)
                samples_mean = np.mean(outputs, axis=1)

                all_outputs.append(outputs)
                all_medians.append(samples_median)
                all_means.append(samples_mean)
                all_targets.append(c_target)
                all_masks.append(target_mask.detach().cpu().numpy())
                all_obs_masks.append(mask.detach().cpu().numpy())
        if i > 0:
            all_medians = np.concatenate(all_medians, axis=0)  # (B*N_B, L_hist, K)
            all_means = np.concatenate(all_means, axis=0)  # (B*N_B, L_hist, K)
            all_targets = np.concatenate(all_targets, axis=0)  # (B*N_B, L_hist, K)
            all_masks = np.concatenate(all_masks, axis=0)  # (B*N_B, L_hist, K)
            all_outputs = np.concatenate(all_outputs, axis=0)  # (B*N_B, n_samples, L_hist, K)
            all_obs_masks = np.concatenate(all_obs_masks, axis=0)  # (B*N_B, L_hist, K)
        else:
            all_medians = all_medians[0]
            all_means = all_means[0]
            all_targets = all_targets[0]
            all_masks = all_masks[0]
            all_outputs = all_outputs[0]
            all_obs_masks = all_obs_masks[0]

        mae_median, _, rmse_median, mape_median, _ = metric(
            all_medians[:,:,reserve_indices], all_targets[:,:,reserve_indices]
        )
        mae_mean, _, rmse_mean, mape_mean, _ = metric(
            all_means[:,:,reserve_indices], all_targets[:,:,reserve_indices]
        )
        CRPS = self._calc_quantile_CRPS(all_targets, all_outputs, all_masks)

        # only visualize for vali dataset
        if setting is None:
            # if setting is None, do not visualize, return directly
            self.model.train()
            return [mae_mean, rmse_mean, mape_mean, mae_median, rmse_median, mape_median, CRPS]

        # starting visualization
        quantiles_imp = self._quantile(all_outputs, all_targets, all_obs_masks)
        #
        available_cols = reserve_indices
        # control the maximum number of subplots to be visualized
        num_subplots = min(len(available_cols), 30)
        available_cols = available_cols[:num_subplots]

        dataind = int(self.L_d / L)

        ncols = 3
        nrows = (num_subplots + ncols - 1) // ncols

        folder_path = "./vali_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # with open(folder_path + "generated_" + str(n_samples) + 'samples_epoch' + str(epoch), 'wb') as f:
        #     pickle.dump(
        #         [
        #             available_cols,
        #             quantiles_imp,
        #             all_targets,
        #             all_masks,
        #             all_obs_masks,
        #         ],
        #         f,
        #     )
        dataset_name = self.args.root_path.split('/')[-1]

        if dataind > 1:
            daily_plot_subplots(
                nrows,
                ncols,
                available_cols,
                L,
                dataind,
                all_means,
                quantiles_imp,
                all_targets,
                all_masks,
                all_obs_masks,
                folder_path,
                epoch,
                dataset_name,
            )
        else:
            plot_subplots(
                nrows,
                ncols,
                available_cols,
                L,
                dataind,
                quantiles_imp,
                all_targets,
                all_masks,
                all_obs_masks,
                folder_path,
                epoch,
            )

        self.model.train()
        return [mae_mean, rmse_mean, mape_mean, mae_median, rmse_median, mape_median, CRPS]
