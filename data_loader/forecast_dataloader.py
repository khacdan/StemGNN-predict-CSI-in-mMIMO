import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd


def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':       # Hàm chuẩn hóa dữ liệu
        if not norm_statistic:              # Nếu chưa có thống kê chuẩn hóa, tính giá trị max và min của dữ liệu
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5    # Tránh chia cho 0 bằng cách thêm giá trị nhỏ
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)  # Giới hạn giá trị trong khoảng [0, 1]
    elif normalize_method == 'z_score':     # Hàm chuẩn hóa dữ liệu z_score
        if not norm_statistic:              # Nếu chưa có thống kê chuẩn hóa, tính giá trị mean và std của dữ liệu
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]# Tránh chia cho 0 bằng cách thay giá trị 0 bằng 1
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic


def de_normalized(data, normalize_method, norm_statistic):
     # Hàm khôi phục dữ liệu từ dạng chuẩn hóa
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data


class ForecastDataset(torch_data.Dataset):
    def __init__(self, df, window_size, horizon, normalize_method=None, norm_statistic=None, interval=1):
        self.window_size = window_size  # Kích thước cửa sổ (window size)
        self.interval = interval        # Khoảng cách giữa các mẫu (interval)
        self.horizon = horizon          # Khoảng thời gian dự báo (horizon)
        self.normalize_method = normalize_method    # Phương pháp chuẩn hóa
        self.norm_statistic = norm_statistic        # Thống kê chuẩn hóa
        df = pd.DataFrame(df)
        # Điền các giá trị thiếu bằng phương pháp forward fill và backward fill
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
#         print("df",df.shape)
        self.df_length = len(df)    # Chiều dài của DataFrame
        self.x_end_idx = self.get_x_end_idx()   # Các chỉ số kết thúc của cửa sổ (window)
#         print(self.x_end_idx)
        if normalize_method:
            self.data, _ = normalized(self.data, normalize_method, norm_statistic)  # Chuẩn hóa dữ liệu nếu cần

    def __getitem__(self, index):
        hi = self.x_end_idx[index]  # Chỉ số kết thúc của cửa sổ
#         print(index)
#         print(hi)
        lo = hi - self.window_size  # Chỉ số bắt đầu của cửa sổ
        train_data = self.data[lo: hi]  # Dữ liệu huấn luyện
#         print("tdata",train_data.shape)
        target_data = self.data[hi:hi + self.horizon]   # Dữ liệu mục tiêu
#         print("tdata",target_data.shape)
        x = torch.from_numpy(train_data).type(torch.float)  # Chuyển đổi dữ liệu huấn luyện thành Tensor
        y = torch.from_numpy(target_data).type(torch.float) # Chuyển đổi dữ liệu mục tiêu thành Tensor
        return x, y

    def __len__(self):
        return len(self.x_end_idx)  # Trả về số lượng mẫu dữ liệu

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        # Mỗi phần tử `hi` trong `x_index_set` là một giới hạn trên để lấy dữ liệu huấn luyện
        # Phạm vi dữ liệu huấn luyện: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
#         print(x_index_set)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        
        return x_end_idx
