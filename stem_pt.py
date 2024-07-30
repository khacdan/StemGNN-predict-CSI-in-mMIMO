import pandas as pd
import numpy as np
import torch
import torch.nn as nn

cr = 512
hori = 9
mode = 5

csv_file_path = 'output/Uma_512/test/predict.csv'
predict = pd.read_csv(csv_file_path)
# 
csv_file_path = 'output/Uma_512/test/target.csv'
target = pd.read_csv(csv_file_path)

predict = torch.tensor(predict.values)
target = torch.tensor(target.values)
# target = targets.to('cpu')
# predict = output_lstm.to('cpu')
print(target.size())
print(predict.size())

torch.save(predict[:,:],f"D:/CMa/{cr}/predict_5_{cr}_3_{hori}.pt")
torch.save(target[:,:],f"D:/CMa/{cr}/target_5_{cr}_3_{hori}.pt")
