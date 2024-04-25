import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.7 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** (epoch  // args.lradj_factor))}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2, alpha=0.7)
    plt.scatter(range(len(true)), true, label='Prediction', s=4)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, alpha=0.7)
        plt.scatter(range(len(preds)), preds, label='Prediction', s=4)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, model_optim, scheduler, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_optim, scheduler, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_optim, scheduler, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_optim, scheduler, path):
        if self.verbose:
            print(f'Validation CRPS decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss
        torch.save({
                    """
                    Save model use model.module.state_dict() instead of model.state_dict()。
                    Without module, the saved model would save the structure of DDP, module
                    would save the model only
                    """
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': model_optim.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    }, path + '/' + 'checkpoint.pth')


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"
    
class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)

def time_features(dates, freq='h'):
    fea_ls = []
    for feat in time_features_from_frequency_str(freq):
        fea_ls.append(feat(dates))
    return np.vstack(fea_ls)


def plot_subplots(
        nrows, 
        ncols, 
        available_cols, 
        L, 
        dataind, 
        quantiles_imp, 
        all_target_np, 
        all_evalpoint_np, 
        all_given_np, 
        path, 
        epoch
        ):
    """
    plot daily subplots by concatenating subseqs into daily seqsk
    """
    plt.rcParams["font.size"] = 20
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(24.0, 3.5*nrows))
    # fig.delaxes(axes[-1][-1])

    for i in range(len(available_cols)):
        k = available_cols[i] # k is the col/feature index
        row = i // ncols
        col = i % ncols
        # all_target_np is of shape (B, L, K)
        df = pd.DataFrame({"x":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_evalpoint_np[dataind,:,k]})
        df = df[df.y != 0]
        df2 = pd.DataFrame({"x":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_given_np[dataind,:,k]})
        df2 = df2[df2.y != 0]

        axes[row][col].plot(range(0,L), quantiles_imp[2][dataind,:,k], color = 'g',linestyle='solid',label='Diff')
        axes[row][col].fill_between(range(0,L), quantiles_imp[0][dataind,:,k],quantiles_imp[4][dataind,:,k],
                        color='g', alpha=0.3)
        axes[row][col].plot(df.x, df.val, color = 'b',marker = 'o', linestyle='None', markersize=2)
        axes[row][col].plot(df2.x,df2.val, color = 'r',marker = 'x', linestyle='None')

        # Get the minimum y-value from the data
        min_y = min(np.min(df.val), np.min(quantiles_imp[0][dataind,:,k]))
        max_y = max(np.max(df.val), np.max(quantiles_imp[4][dataind,:,k]))
        if min_y > 0:
            bottom = min_y - min_y*0.1
        else:
            bottom = min_y + min_y*0.1
            
        axes[row][col].set_ylim(bottom= bottom)  # Set the y-axis lower limit
        axes[row][col].set_ylim(top= max_y + max_y*0.1)  # Set the y-axis upper limit

        # axes[row][col].lengend()

        if col == 0:
            plt.setp(axes[row, 0], ylabel='Traffic Flow)')
        if row == nrows-1:
            plt.setp(axes[-1, col], xlabel='Time')

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
    plt.savefig(f"{path}epoch({epoch}).png", dpi=300)
    plt.close()


def daily_plot_subplots(
        nrows, 
        ncols, 
        available_cols, 
        L, 
        dataind, 
        appro_mean,
        quantiles_imp, 
        all_target_np, 
        all_evalpoint_np, 
        all_given_np, 
        path, 
        epoch,
        dataset_name
        ):
    """
    plot daily subplots by concatenating subseqs into daily seqsk
    """
    plt.rcParams["font.size"] = 20
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(24.0, 3.5*nrows))
    # fig.delaxes(axes[-1][-1])

    for i in range(len(available_cols)):
        k = available_cols[i] # k is the col/feature index
        row = i // ncols
        col = i % ncols
        # all_target_np is of shape (B, L, K)
        df = pd.DataFrame({"x":np.arange(0,L*dataind), "val":all_target_np[:dataind,:,k].reshape(-1), "y":all_evalpoint_np[:dataind,:,k].reshape(-1)})
        df = df[df.y != 0]
        df2 = pd.DataFrame({"x":np.arange(0,L*dataind), "val":all_target_np[:dataind,:,k].reshape(-1), "y":all_given_np[:dataind,:,k].reshape(-1)})
        df2 = df2[df2.y != 0]

        axes[row][col].plot(range(0,L*dataind), quantiles_imp[2][:dataind,:,k].reshape(-1), color = 'g',linestyle='solid',label='Approx.Median')
        axes[row][col].plot(range(0,L*dataind), appro_mean[:dataind,:,k].reshape(-1), color = 'r',linestyle='solid',label='Approx.Mean')
        axes[row][col].fill_between(range(0,L*dataind), quantiles_imp[0][:dataind,:,k].reshape(-1),quantiles_imp[4][:dataind,:,k].reshape(-1),
                        color='g', alpha=0.3,label='90% CI')
        axes[row][col].plot(df.x, df.val, color = 'b',marker = 'o', linestyle='None', markersize=2, label='Ground Truth')
        axes[row][col].plot(df2.x,df2.val, color = 'r',marker = 'x', linestyle='None')

        # Get the minimum y-value from the data
        min_y = min(np.min(df.val), np.min(quantiles_imp[0][:dataind,:,k].reshape(-1)))
        max_y = max(np.max(df.val), np.max(quantiles_imp[4][:dataind,:,k].reshape(-1)))
        if min_y > 0:
            bottom = min_y - min_y*0.1
        else:
            bottom = min_y + min_y*0.1
            
        axes[row][col].set_ylim(bottom= bottom)  # Set the y-axis lower limit
        axes[row][col].set_ylim(top= max_y + max_y*0.1)  # Set the y-axis upper limit

        # axes[row][col].lengend()

        if col == 0:
            if dataset_name in ['PeMS7_228', 'PeMS7_1026','Seattle']:
                plt.setp(axes[row, 0], ylabel='Traffic Speed (mph))')
        if row == nrows-1:
            plt.setp(axes[-1, col], xlabel='Time')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
    plt.savefig(f"{path}epoch({epoch}).png", dpi=300)
    plt.close()