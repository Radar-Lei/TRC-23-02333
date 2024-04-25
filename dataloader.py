from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from utils import time_features
import numpy as np
import datetime

class Dataset_Custom(Dataset):
    def __init__(self, 
                 root_path, # the path of the dataset
                 flag='train', 
                 seq_len=None,
                 standardaization=True, # whether to standardize the data
                 timeenc=0, # type of time encoding
                 freq='t', 
                 data_shrink = 1, # time interval to segment the time series, larger number decreases sample size
                 fixed_seed= 20
                 ):
        # info
        if seq_len == None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = seq_len
        # init
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        if root_path in ['./dataset/PeMS7_228','./dataset/PeMS7_1026','./dataset/Seattle']:
            self.L_d = 288

        self.standardaization = standardaization
        self.timeenc = timeenc
        self.freq = freq
        self.data_shrink = data_shrink
        self.fixed_seed = fixed_seed

        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        self.standardizer = StandardScaler()
           

        if self.root_path == './dataset/PeMS7_228':
            df_raw = pd.read_csv(os.path.join(self.root_path, 'PeMSD7_V_228.csv'), header=None)
            datetime_range = pd.date_range(start='2012-05-01', end='2012-06-30', freq='5min')
            # Filter out the weekends
            datetime_range = datetime_range[datetime_range.to_series().dt.weekday < 5]            
            df_raw['date'] = datetime_range

            self.num_day = 44 # only weekdays
            
            weight_A = pd.read_csv(os.path.join(self.root_path, 'PeMSD7_W_228.csv'), header=None).values
            weight_A_norm = (weight_A - weight_A.mean()) / weight_A.std()
            self.spatial_inp = weight_A_norm
            
        elif self.root_path == './dataset/PeMS7_1026':
            df_raw = pd.read_csv(os.path.join(self.root_path, 'PeMSD7_V_1026.csv'), header=None)
            datetime_range = pd.date_range(start='2012-05-01', end='2012-06-30', freq='5min')
            
            datetime_range = datetime_range[datetime_range.to_series().dt.weekday < 5]            
            df_raw['date'] = datetime_range
            
            self.num_day = 44 # only weekdays
            
            weight_A = pd.read_csv(os.path.join(self.root_path, 'PeMSD7_W_1026.csv'), header=None).values
            weight_A_norm = (weight_A - weight_A.mean()) / weight_A.std()
            self.spatial_inp = weight_A_norm
            
        elif self.root_path == './dataset/Seattle':
            df_raw = pd.read_pickle('./dataset/Seattle/speed_matrix_2015') # (D*L_d, K)
            datetime_range = pd.date_range(start='2015/01/01', end='2016/01/01', freq='5min')[:-1]
            
            self.num_day = 365
            
            # the sensor id's in data_arr_df actually has duplicated sensor ids
            location_info = pd.read_csv('./dataset/Seattle/Cabinet Location Information.csv')
            distance_df = pd.DataFrame({'SensorName': [col[1:] for col in df_raw.columns]})
            dist_merged_df = pd.merge(distance_df, location_info[['CabName', 'Lat', 'Lon']], left_on='SensorName', right_on='CabName', how='left')

            # drop the redundant 'CabName' column
            dist_merged_df = dist_merged_df.drop('CabName', axis=1)

            # Compute the adjacency matrix
            adj_matrix = np.zeros((len(dist_merged_df), len(dist_merged_df)))
            for i in range(len(dist_merged_df)):
                for j in range(i+1, len(dist_merged_df)):
                    lat1, lon1 = dist_merged_df.iloc[i]['Lat'], dist_merged_df.iloc[i]['Lon']
                    lat2, lon2 = dist_merged_df.iloc[j]['Lat'], dist_merged_df.iloc[j]['Lon']
                    dist = self._haversine(lat1, lon1, lat2, lon2)
                    adj_matrix[i,j] = dist
                    adj_matrix[j,i] = dist

            weight_A_norm = (adj_matrix - adj_matrix.mean()) / adj_matrix.std()
            self.spatial_inp = weight_A_norm
            df_raw['date'] = datetime_range   
            
        cols = list(df_raw.columns)
        # cols.remove(cols[-1])
        cols.remove('date')

        df_data = df_raw[cols].values

        num_days_test = 2
        num_days_vali = 2
        num_days_train = self.num_day - num_days_test - num_days_vali

        df_data = pd.DataFrame(df_data,index=pd.DatetimeIndex(df_raw['date'].values))

        # Get the start and end dates from df_data index
        start_date = df_data.index.min()
        end_date = df_data.index.max()

        # Generate a range of dates between start_date and end_date
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        if self.root_path in ['./dataset/PeMS7_228','./dataset/PeMS7_1026']:
            # PeMS7 only contains data for weekdays
            date_range = date_range[date_range.to_series().dt.weekday < 5]
        elif self.root_path in ['./dataset/Seattle']:
            # Seattle contains data for all days
            date_range = date_range

        # structurally missing can use all dates for training
        np.random.seed(self.fixed_seed)
        
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

        # Convert to datetime
        train_dates = train_dates.astype('datetime64[D]').astype(datetime.datetime)
        vali_dates = vali_dates.astype('datetime64[D]').astype(datetime.datetime)
        test_dates = test_dates.astype('datetime64[D]').astype(datetime.datetime)

        # Filter df_data based on the selected dates
        df_train = df_data.copy(deep=True)
        df_vali = df_data[pd.Series(df_data.index.date, index=df_data.index).isin(vali_dates)]
        df_test = df_data[pd.Series(df_data.index.date, index=df_data.index).isin(test_dates)]

        # the following obtain the actual mask for the data
        train_act_mask = np.where(df_train==0, 0, 1)
        vali_act_mask = np.where(df_vali==0, 0, 1)
        test_act_mask = np.where(df_test==0, 0, 1)

        if self.standardaization:
            self.standardizer.fit(df_train.values)
            train_data = self.standardizer.transform(df_train.values)
            vali_data = self.standardizer.transform(df_vali.values)
            test_data = self.standardizer.transform(df_test.values)
        else:
            train_data = df_train.values
            vali_data = df_vali.values
            test_data = df_test.values

        if self.set_type == 0:
            df_stamp = df_train.reset_index().rename(columns={'index': 'date'})[['date']]
        elif self.set_type == 1:
            df_stamp = df_vali.reset_index().rename(columns={'index': 'date'})[['date']]
        elif self.set_type == 2:
            df_stamp = df_test.reset_index().rename(columns={'index': 'date'})[['date']]
        else:
            df_stamp = df_data.reset_index().rename(columns={'index': 'date'})[['date']]

        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(labels='date', axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.set_type == 0:
            self.data_x = train_data
            self.data_y = train_data
            self.actual_mask = train_act_mask
            self.curr_num_days = len(train_dates)
        elif self.set_type == 1:
            self.data_x = vali_data
            self.data_y = vali_data
            self.actual_mask = vali_act_mask
            self.curr_num_days = len(vali_dates)
        elif self.set_type == 2:
            self.data_x = test_data
            self.data_y = test_data
            self.actual_mask = test_act_mask
            self.curr_num_days = len(test_dates)

        # prevent the sampled sequence to be across two days
        if self.set_type == 0:
            self.valid_indices = [i for i in range(self.L_d * self.curr_num_days) if i % self.L_d 
                                <= (self.L_d - self.seq_len)][::self.data_shrink]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.set_type == 0: # when not pred
            s_begin = self.valid_indices[index]
            s_end = s_begin + self.seq_len

        elif (self.set_type == 1) or (self.set_type == 2):
            s_begin = index * self.seq_len
            s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        return seq_x, seq_x_mark, self.actual_mask[s_begin:s_end], self.spatial_inp

    def __len__(self):
        if self.set_type == 0:
            return len(self.valid_indices)
        else: # self.set_type == 3: # pred
            return int(len(self.data_x) / self.seq_len)

    def inverse_transform(self, data):
        return self.standardizer.inverse_transform(data)
    
    def __var__(self):
        return self.standardizer.var_
    
    def __mean__(self):
        return self.standardizer.mean_
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = R * c
        return d
    

def data_provider(args, flag):
    Data = Dataset_Custom
    timeenc = 0 if args.embed != 'timeF' else 1

    if (flag == 'pred'):
        shuffle_flag = False
        drop_last = True
        
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    root_path = args.root_path

    data_set = Data(
        root_path=root_path,
        flag=flag,
        seq_len=args.seq_len,
        timeenc=timeenc,
        freq=args.freq,
        data_shrink=args.data_shrink,
        fixed_seed=args.fixed_seed,
    )
    
    print(flag, len(data_set))
    
    if flag == 'val' or flag == 'test':
        shuffle_flag = False
        if len(data_set) < batch_size:
            batch_size = len(data_set)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader