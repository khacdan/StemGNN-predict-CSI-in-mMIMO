import numpy as np


def masked_MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.   # Tham số v là giá trị thực
    :param v_: np.ndarray or int, prediction.    # Tham số v_ là giá trị dự đoán
    :param axis: axis to do calculation.         # Tham số axis để xác định trục tính toán
    :return: int, MAPE averages on all elements of input.   # Trả về giá trị MAPE trung bình trên tất cả các phần tử đầu vào
    '''
    mask = (v == 0) # Tạo mặt nạ cho các giá trị v bằng 0
    percentage = np.abs(v_ - v) / np.abs(v) # Tính phần trăm sai số tuyệt đối
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)  # mask the dividing-zero as invalid
        result = masked_array.mean(axis=axis)   # Tính trung bình trên các phần tử hợp lệ
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)    # Nếu kết quả là một Mảng mặt nạ, thay thế các giá trị bị che bằng NaN
        else:
            return result   # Trả về kết quả
    return np.mean(percentage, axis).astype(np.float64) # Tính trung bình phần trăm sai số tuyệt đối và chuyển đổi thành float64


def MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.  # Tham số v là giá trị thực
    :param v_: np.ndarray or int, prediction.   # Tham số v_ là giá trị dự đoán
    :param axis: axis to do calculation.        # Tham số axis để xác định trục tính toán
    :return: int, MAPE averages on all elements of input.   # Trả về giá trị MAPE trung bình trên tất cả các phần tử đầu vào
    '''
    mape = (np.abs(v_ - v) / np.abs(v)+1e-5).astype(np.float64) # Tính phần trăm sai số tuyệt đối và thêm một giá trị nhỏ để tránh chia cho 0
    mape = np.where(mape > 5, 5, mape)  # Giới hạn giá trị MAPE tối đa là 5
    return np.mean(mape, axis)  # Tính trung bình MAPE trên trục chỉ định


def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.   # Trả về giá trị RMSE trung bình trên tất cả các phần tử đầu vào
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64) # Tính căn bậc hai của trung bình bình phương sai số


def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.    # Trả về giá trị MAE trung bình trên tất cả các phần tử đầu vào
    '''
    return np.mean(np.abs(v_ - v), axis).astype(np.float64) # Tính trung bình sai số tuyệt đối



def evaluate(y, y_hat, by_step=False, by_node=False):
    '''
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    '''
    if not by_step and not by_node:
        return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)# Đánh giá toàn bộ dữ liệu
    if by_step and by_node:
        return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)# Đánh giá theo từng bước thời gian và từng nút
    if by_step:
        return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))# Đánh giá theo từng bước thời gian
    if by_node:
        return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))# Đánh giá theo từng nút
