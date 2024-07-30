import json
from datetime import datetime

from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from models.base_model import Model
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import time
import os

from utils.math_utils import evaluate


def save_model(model, model_dir, epoch=None):
    if model_dir is None:   # Kiểm tra nếu đường dẫn thư mục là None thì không làm gì cả
        return
    if not os.path.exists(model_dir):   # Nếu thư mục không tồn tại, tạo mới
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else '' # Chuyển đổi epoch thành chuỗi nếu không None
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')  # Tạo tên file lưu trữ
    with open(file_name, 'wb') as f:    # Mở file ở chế độ ghi nhị phân
        torch.save(model, f)            # Lưu mô hình sử dụng torch.save

# Hàm tải mô hình
def load_model(model_dir, epoch=None):  # Kiểm tra nếu đường dẫn thư mục là None thì không làm gì cả
    if not model_dir:
        return
    epoch = str(epoch) if epoch else '' # Chuyển đổi epoch thành chuỗi nếu không None
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')  # Tạo tên file cần tải
    if not os.path.exists(model_dir):    # Nếu thư mục không tồn tại, tạo mới
        os.makedirs(model_dir)
    if not os.path.exists(file_name):     # Nếu file không tồn tại, trả về None
        return
    with open(file_name, 'rb') as f:        # Mở file ở chế độ đọc nhị phân
        model = torch.load(f)               # Tải mô hình sử dụng torch.load
    return model

# Hàm suy luận (dự đoán) trên tập dữ liệu
def inference(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()    # Đặt mô hình vào chế độ đánh giá (không huấn luyện)
    with torch.no_grad():   # Không tính toán gradient
        for i, (inputs, target) in enumerate(dataloader):   # Duyệt qua các batch dữ liệu trong dataloader
            inputs = inputs.to(device)           #Chuyển inputs lên thiết bị (CPU/GPU)
            target = target.to(device)          # Chuyển target lên thiết bị
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=float)   # Khởi tạo mảng dự đoán
            while step < horizon:  # Lặp qua từng bước dự đoán
                forecast_result, a = model(inputs)  # Dự đoán từ mô hình
                len_model_output = forecast_result.size()[1]    # Độ dài của kết quả dự đoán
                if len_model_output == 0:       # Nếu kết quả rỗng, báo lỗi
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                       :].clone()

                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                step += min(horizon - step, len_model_output)   # Tăng bước dự đoán
            forecast_set.append(forecast_steps)                  # Thêm kết quả dự đoán vào forecast_set
            target_set.append(target.detach().cpu().numpy())       # Thêm target vào target_set
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0) # Trả về kết quả dự đoán và target

# Hàm xác thực mô hình trên tập dữ liệu
def validate(model, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon,
             result_file=None):
    start = datetime.now()
    forecast_norm, target_norm = inference(model, dataloader, device,
                                           node_cnt, window_size, horizon)  # Gọi hàm inference để dự đoán
#     print("forecast", forecast_norm)
#     print("target", target_norm)
    if normalize_method and statistic:      # Nếu có phương pháp và số liệu chuẩn hóa
        forecast = de_normalized(forecast_norm, normalize_method, statistic)    # Giải chuẩn hóa dữ liệu dự đoán
        target = de_normalized(target_norm, normalize_method, statistic)        # Giải chuẩn hóa target
    else:
        forecast, target = forecast_norm, target_norm                   # Nếu không có thì giữ nguyên
    score = evaluate(target, forecast)      # Đánh giá kết quả dự đoán
    score_by_node = evaluate(target, forecast, by_node=True)        # Đánh giá theo từng nút
    end = datetime.now()

    score_norm = evaluate(target_norm, forecast_norm)    # Đánh giá kết quả chuẩn hóa
    print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    if result_file:     # Nếu có file kết quả
        if not os.path.exists(result_file): # Nếu thư mục không tồn tại, tạo mới
            os.makedirs(result_file)
        step_to_print = 0
        forcasting_2d = forecast[:, step_to_print, :]
        forcasting_2d_target = target[:, step_to_print, :]

        np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
        np.savetxt(f'{result_file}/predict_abs_error.csv',
                   np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
        np.savetxt(f'{result_file}/predict_ape.csv',
                   np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

    return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                rmse=score[2], rmse_node=score_by_node[2])  # Trả về kết quả đánh giá

# Hàm huấn luyện mô hình
def train(train_data, valid_data, args, result_file):
    node_cnt = train_data.shape[1]  # Số lượng nút (nodes) trong dữ liệu huấn luyện
    model = Model(node_cnt, 2, args.window_size, args.multi_layer, horizon=args.horizon)    # Khởi tạo mô hình
    model.to(args.device)   # Chuyển mô hình lên thiết bị (CPU/GPU)
    if len(train_data) == 0:    # Kiểm tra dữ liệu huấn luyện có đủ không
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:    # Kiểm tra dữ liệu xác thực có đủ không
        raise Exception('Cannot organize enough validation data')

    if args.norm_method == 'z_score':   # Chuẩn hóa dữ liệu theo z-score
        train_mean = np.mean(train_data, axis=0)
#         print(np.shape(train_mean))
        train_std = np.std(train_data, axis=0)
#         print(np.shape(train_std))
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max': # Chuẩn hóa dữ liệu theo min-max
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:   # Nếu có thông số chuẩn hóa, lưu vào file json
            json.dump(normalize_statistic, f)

    if args.optimizer == 'RMSProp': # Khởi tạo optimizer
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
#     print("train data", train_data.shape)
    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)  # Tạo dataset huấn luyện
#     print("train set", train_set)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)   # Tạo dataset xác thực
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                         num_workers=0) # Tạo dataloader huấn luyện
#     print("loader", train_loader)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)  # Tạo dataloader xác thực
############################
    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)    # Khởi tạo hàm mất mát
    # Tính tổng số lượng tham số có thể huấn luyện của mô hình
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue    # Bỏ qua các tham số không cần tính toán gradient
        param = parameter.numel()   # Số lượng phần tử trong tham số
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    best_validate_mae = np.inf      # Khởi tạo giá trị MAE tốt nhất
    validate_score_non_decrease_count = 0   # Đếm số lần giá trị MAE không giảm
    performance_metrics = {}        # Khởi tạo dictionary để lưu các chỉ số hiệu suất
    for epoch in range(args.epoch): # Lặp qua từng epoch
#         print("epoch",epoch)      
        epoch_start_time = time.time()       # Lấy thời gian bắt đầu epoch
        model.train()       # Đặt mô hình vào chế độ huấn luyện     
        loss_total = 0  # Tổng mất mát cho epoch
        cnt = 0 # Đếm số batch

        for i, (inputs, target) in enumerate(train_loader): # Duyệt qua các batch trong dataloader huấn luyện
#             print("input shape",inputs.shape)
#             print("target shape",target.shape)
#             print(i)
            inputs = inputs.to(args.device) # Chuyển inputs lên thiết bị (CPU/GPU)
            target = target.to(args.device) # Chuyển target lên thiết bị (CPU/GPU)
            model.zero_grad()               # Đặt lại gradient của mô hình
            forecast, _ = model(inputs)         # Dự đoán từ mô hình
#             if epoch > 48:
#                 print("forecast", forecast)
#                 print("target", target)
#             print("forecast size", forecast.size())
#             print("target size", target.size())
            loss = forecast_loss(forecast, target)  # Tính toán mất mát (loss)
            cnt += 1        # Tăng đếm số batch
            loss.backward() # Lan truyền ngược gradient
            my_optim.step() # Cập nhật trọng số mô hình
            loss_total += float(loss)   # Cộng dồn mất mát
        # In ra thông tin sau mỗi epoch
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt))
        save_model(model, result_file, epoch)   # Lưu mô hình sau mỗi epoch
        if (epoch+1) % args.exponential_decay_step == 0:    
            my_lr_scheduler.step()  # Cập nhật learning rate theo scheduler
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validate(model, valid_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         result_file=result_file)   # Xác thực mô hình trên tập xác thực
            if best_validate_mae > performance_metrics['mae']:  # Kiểm tra nếu MAE tốt hơn
                best_validate_mae = performance_metrics['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
            # save model
            if is_best_for_now:
                save_model(model, result_file)  # Lưu mô hình nếu tốt nhất
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break   # Dừng sớm nếu số lần MAE không giảm lớn hơn ngưỡng cho phép
    return performance_metrics, normalize_statistic # Trả về các chỉ số hiệu suất và thông số chuẩn hóa

# Hàm kiểm thử mô hình
def test(test_data, args, result_train_file, result_test_file):
    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)  # Đọc thông số chuẩn hóa từ file json
    model = load_model(result_train_file)   # Tải mô hình đã huấn luyện
    node_cnt = test_data.shape[1]           # Số lượng nút trong dữ liệu kiểm thử
    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic)   # Tạo dataset kiểm thử
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)               # Tạo dataloader kiểm thử
    performance_metrics = validate(model, test_loader, args.device, args.norm_method, normalize_statistic,
                      node_cnt, args.window_size, args.horizon,  
                      result_file=result_test_file)      # Xác thực mô hình trên tập kiểm thử
    mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
    # In ra các chỉ số hiệu suất trên tập kiểm thử
    print('Performance on test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape, mae, rmse))