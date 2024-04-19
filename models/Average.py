"""
Linear Interpolation

This baseline imputes the missing values of sensor-free nodes using the mean traffic states of 
the neighboring sensor-equipped nodes.

bash ./scripts/Linear_interpolation.sh > $(date +'%y%m%d-%H%M%S')_Linear_interpolation_log.txt
"""
import numpy as np
import pandas as pd
import os
import datetime
import argparse

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

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

def data_loader(data_path, seed=42):
    if data_path in ['./dataset/PeMS7_228','./dataset/PeMS7_1026']:
        if data_path == './dataset/PeMS7_228':
            df_raw = pd.read_csv(os.path.join(data_path, 'PeMSD7_V_228.csv'), header=None)
            adj_mat = pd.read_csv(os.path.join(data_path, 'PeMSD7_W_228.csv'), header=None).values
        elif data_path == './dataset/PeMS7_1026':
            df_raw = pd.read_csv(os.path.join(data_path, 'PeMSD7_V_1026.csv'), header=None)
            adj_mat = pd.read_csv(os.path.join(data_path, 'PeMSD7_W_1026.csv'), header=None).values
        date_range = pd.date_range(start='2012-05-01', end='2012-06-30', freq='D')
        # Filter out the weekends
        date_range = date_range[date_range.to_series().dt.weekday < 5]
        datetime_range = pd.date_range(start='2012-05-01', end='2012-06-30', freq='5min')
        # Filter out the weekends
        datetime_range = datetime_range[datetime_range.to_series().dt.weekday < 5]            
        df_raw['date'] = datetime_range        
        num_day = 44 # only weekdays
        
    elif data_path == './dataset/Seattle':
        df_raw = pd.read_pickle('./dataset/Seattle/speed_matrix_2015') # (D*L_d, K)
        date_range = pd.date_range(start='2015/01/01', end='2016/01/01', freq='D')[:-1]
        datetime_range = pd.date_range(start='2015/01/01', end='2016/01/01', freq='5min')[:-1]
        num_day = 365

        # the sensor id's in data_arr_df actually has duplicated sensor ids
        location_info = pd.read_csv('./dataset/Seattle/Cabinet Location Information.csv')
        distance_df = pd.DataFrame({'SensorName': [col[1:] for col in df_raw.columns]})
        dist_merged_df = pd.merge(distance_df, location_info[['CabName', 'Lat', 'Lon']], left_on='SensorName', right_on='CabName', how='left')

        # drop the redundant 'CabName' column
        dist_merged_df = dist_merged_df.drop('CabName', axis=1)

        # Compute the adjacency matrix
        adj_mat = np.zeros((len(dist_merged_df), len(dist_merged_df)))
        for i in range(len(dist_merged_df)):
            for j in range(i+1, len(dist_merged_df)):
                lat1, lon1 = dist_merged_df.iloc[i]['Lat'], dist_merged_df.iloc[i]['Lon']
                lat2, lon2 = dist_merged_df.iloc[j]['Lat'], dist_merged_df.iloc[j]['Lon']
                dist = _haversine(lat1, lon1, lat2, lon2)
                adj_mat[i,j] = dist
                adj_mat[j,i] = dist
        df_raw['date'] = datetime_range  
        
    cols = list(df_raw.columns)
    cols.remove('date')
    df_data = df_raw[cols].values
    df_data = pd.DataFrame(df_data,index=pd.DatetimeIndex(df_raw['date'].values))
    
    num_days_test = 2
    num_days_vali = 2
    num_days_train = num_day - num_days_test - num_days_vali
    np.random.seed(seed)
    # Create a list of all available dates
    available_dates = date_range.tolist()

    # Select random dates for the training set
    train_dates = np.random.choice(available_dates, size=num_days_train, replace=False)
    # Remove these dates from the list of available dates
    available_dates = [date for date in available_dates if date not in train_dates]

    # Select random dates for the validation set
    vali_dates = np.random.choice(available_dates, size=num_days_vali, replace=False)
    # Remove these dates from the list of available dates
    available_dates = [date for date in available_dates if date not in vali_dates]

    # Select random dates for the test set
    test_dates = np.random.choice(available_dates, size=num_days_test, replace=False)
    
    test_dates = test_dates.astype('datetime64[D]').astype(datetime.datetime)

    df_test = df_data[pd.Series(df_data.index.date, index=df_data.index).isin(test_dates)]
    
    return df_test.values, adj_mat

def imputation(arr_test, adj_mat, num_neighbor, missing_rate, seed):
    _, K = arr_test.shape
    np.random.seed(seed)
    # presumed sensor-free nodes
    reserve_indices = np.random.choice(
        range(K), round(K * missing_rate), replace=False
        )
    sensor_equipped_node_ids = [i for i in range(K) if i not in reserve_indices]
    
    arr_imputed_test = np.zeros_like(arr_test)
    
    for node in reserve_indices:
        distances = adj_mat[node]
        node_distances = [(i, distances[i]) for i in sensor_equipped_node_ids]
        node_distances.sort(key=lambda x: x[1])
        closest_nodes = [node_id for node_id, _ in node_distances[:num_neighbor]]
        arr_imputed_test[:, node] = np.mean(arr_test[:, closest_nodes], axis=1)
    
    mae, _, rmse, mape, _ = metric(arr_imputed_test[:,reserve_indices], arr_test[:,reserve_indices])
    
    print('Imputation MAE: {:.2f}'.format(mae))
    print('Imputation RMSE: {:.2f}'.format(rmse))
    print('Imputation MAPE: {:.2f}'.format(mape))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Interpolation')
    # './dataset/Seattle', './dataset/PeMS7_228', './dataset/PeMS7_1026'
    parser.add_argument('--data_path', type=str, default='./dataset/PeMS7_1026', help='data path')
    parser.add_argument('--num_neighbor', type=int, default=3, help='number of neighbors')
    parser.add_argument('--missing_rate', type=float, default=0.3, help='missing rate') 
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    print("Linear Interpolation: {}".format(args.data_path.split('/')[-1]))
    
    print("Misssing Rate: {}".format(args.missing_rate))
    arr_test, adj_mat = data_loader(args.data_path, args.seed)
    imputation(arr_test, adj_mat, args.num_neighbor, args.missing_rate, args.seed)
    print("=====================================")