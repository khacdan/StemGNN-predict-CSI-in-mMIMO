import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.handler import train, test
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)# Training
parser.add_argument('--evaluate', type=bool, default=True)# Đánh giá
parser.add_argument('--dataset', type=str, default='Uma_128')# Load data
parser.add_argument('--window_size', type=int, default=8) #Dấu thời gian lịch sử
parser.add_argument('--horizon', type=int, default=7)# Khoảng thời gian dự báo
parser.add_argument('--train_length', type=float, default=7) 
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=1)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--multi_layer', type=int, default=5)# Số layer
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--validate_freq', type=int, default=1)# Tần suất xác thực
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--norm_method', type=str, default='z_score')# Pp chuẩn hóa
parser.add_argument('--optimizer', type=str, default='RMSprop')# Trình tối ưu hóa
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)#
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)# Tỷ lệ drop out
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)

# X = torch.load("F:\CSIP\X_512.pt")
# H = torch.load("F:\CSIP\H_512.pt")
# X = X.numpy()
# H = H.numpy()
# # X = X[:,:,0]
# # H = H[:,:,0]
# X = pd.DataFrame(X) 
# X.to_csv("C:\\Users\\ronal\\OneDrive\\Desktop\\StemGNN-master\\dataset\\Uma_512.csv",index=False) 

#Then, to reload:
# df = pd.read_csv("testfile")



args = parser.parse_args()
print(f'Training configs: {args}')
data_file = os.path.join('dataset', args.dataset + '.csv')  # Đường dẫn tới tập dữ liệu
result_train_file = os.path.join('output', args.dataset, 'train')   # Đường dẫn lưu kết quả huấn luyện
result_test_file = os.path.join('output', args.dataset, 'test')     # Đường dẫn lưu kết quả kiểm tra
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)  # Tạo thư mục
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)   # Tạo thư mục
data = pd.read_csv(data_file).values    # Đọc dữ liệu từ file CSV và chuyển thành mảng numpy
print("shape:",data.shape)
# split data
# Chia dữ liệu thành các phần huấn luyện, xác thực và kiểm tra
train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[:int(train_ratio * len(data))]
valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
test_data = data[int((train_ratio + valid_ratio) * len(data)):]
# print(test_data[0])
# tes = H[int((train_ratio + valid_ratio) * len(data)):]
# print(tes[0])
torch.manual_seed(0)    # Đặt seed để kết quả có thể tái tạo
if __name__ == '__main__':
    if args.train:
        try:
            before_train = datetime.now().timestamp()   # Lấy thời gian trước khi huấn luyện
            _, normalize_statistic = train(train_data, valid_data, args, result_train_file)     # Huấn luyện mô hình
            after_train = datetime.now().timestamp()    # Lấy thời gian sau khi huấn luyện
            print(f'Training took {(after_train - before_train) / 60} minutes') # Tính thời gian huấn luyện
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')    # Thông báo nếu huấn luyện bị dừng sớm
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        test(test_data, args, result_train_file, result_test_file)  # Đánh giá mô hình
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')# Tính thời gian đánh giá
    print('done')
