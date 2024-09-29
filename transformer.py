# ========== Import Modules ========== #
import copy
import datetime
import logging
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import shioaji as sj
import talib as ta
import torch
import wandb

from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import KFold
from tqdm import tqdm
# ================================================== #


# ========== Hyper parameters and parameters ========== #
## Hyperparameters
SEQ_LEN = int(90)
BATCH_SIZE = int(64)
EPOCHS = 20
D_MODEL = 24
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
DROPOUT_RATE = 0.1
LEARNING_RATE = 1e-5
K = 5
PATIENCE = 3
MIN_DELTA_PERCENTAGE = 0.01

## Parameters
MIN_LATER = 30  # The minute we want to predict in the future
DEVICE = (torch.cuda.is_available() and 'cuda:0') or 'cpu'

BEGIN_TIME = datetime.time(9, 15)
END_TIME = datetime.time(13, 15)

UNIT_PCR = 0.001 # Unit price change rate
TQDM_LEAVE = True
WANDB_RELATED = False

FEATURE_NUM = None  # Will be updated later

LOG_FILE_PATH = './log/transformer2.log'
# ================================================== #


# ========== Classes ========== #
## Dataset and DataLoader
class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, begin, end):
        self.TXF = torch.load('./tensor/TXF.pt', weights_only = True).double()
        self.TXF_close = torch.load('./tensor/TXF_close.pt', weights_only = True).double()
        self.bias_short = torch.load('./tensor/bias_short.pt', weights_only = True).double()
        self.bias_long = torch.load('./tensor/bias_long.pt', weights_only = True).double()
        self.adosc = torch.load('./tensor/adosc.pt', weights_only = True).double()
        self.bband_upper = torch.load('./tensor/bband_upper.pt', weights_only = True).double()
        self.bband_middle = torch.load('./tensor/bband_middle.pt', weights_only = True).double()
        self.bband_lower = torch.load('./tensor/bband_lower.pt', weights_only = True).double()
        self.keltner_upper = torch.load('./tensor/keltner_upper.pt', weights_only = True).double()
        self.keltner_middle = torch.load('./tensor/keltner_middle.pt', weights_only = True).double()
        self.keltner_lower = torch.load('./tensor/keltner_lower.pt', weights_only = True).double()
        self.K = torch.load('./tensor/K.pt', weights_only = True).double()
        self.D = torch.load('./tensor/D.pt', weights_only = True).double()
        self.J = torch.load('./tensor/J.pt', weights_only = True).double()
        self.solid_kbar_diff = torch.load('./tensor/solid_kbar_diff.pt', weights_only = True).double()
        self.upper_shadow = torch.load('./tensor/upper_shadow.pt', weights_only = True).double()
        self.lower_shadow = torch.load('./tensor/lower_shadow.pt', weights_only = True).double()
        self.kbar_diff = torch.load('./tensor/kbar_diff.pt', weights_only = True).double()
        self.cci_short = torch.load('./tensor/cci_short.pt', weights_only = True).double()
        self.cci_long = torch.load('./tensor/cci_long.pt', weights_only = True).double()
        self.solid_kbar_diff = torch.load('./tensor/solid_kbar_diff.pt', weights_only = True).double()
        self.upper_shadow = torch.load('./tensor/upper_shadow.pt', weights_only = True).double()
        self.lower_shadow  = torch.load('./tensor/lower_shadow.pt', weights_only = True).double()
        self.kbar_diff = torch.load('./tensor/kbar_diff.pt', weights_only = True).double()

        self.raw_close_price = torch.load('./tensor/TXF_close.pt', weights_only = True).double()

        self.TXF = torch.nn.functional.normalize(self.TXF, dim = 0)
        self.TXF_close = torch.nn.functional.normalize(self.TXF_close, dim = 0)
        self.bias_short = torch.nn.functional.normalize(self.bias_short, dim = 0)
        self.bias_long = torch.nn.functional.normalize(self.bias_long, dim = 0)
        self.adosc = torch.nn.functional.normalize(self.adosc, dim = 0)
        self.bband_upper = torch.nn.functional.normalize(self.bband_upper, dim = 0)
        self.bband_middle = torch.nn.functional.normalize(self.bband_middle, dim = 0)
        self.bband_lower = torch.nn.functional.normalize(self.bband_lower, dim = 0)
        self.keltner_upper = torch.nn.functional.normalize(self.keltner_upper, dim = 0)
        self.keltner_middle = torch.nn.functional.normalize(self.keltner_middle, dim = 0)
        self.keltner_lower = torch.nn.functional.normalize(self.keltner_lower, dim = 0)
        self.K = torch.nn.functional.normalize(self.K, dim = 0)
        self.D = torch.nn.functional.normalize(self.D, dim = 0)
        self.J = torch.nn.functional.normalize(self.J, dim = 0)
        self.solid_kbar_diff = torch.nn.functional.normalize(self.solid_kbar_diff, dim = 0)
        self.upper_shadow = torch.nn.functional.normalize(self.upper_shadow, dim = 0)
        self.lower_shadow = torch.nn.functional.normalize(self.lower_shadow, dim = 0)
        self.kbar_diff = torch.nn.functional.normalize(self.kbar_diff, dim = 0)
        self.cci_short = torch.nn.functional.normalize(self.cci_short, dim = 0)
        self.cci_long = torch.nn.functional.normalize(self.cci_long, dim = 0)

        self.data_len = end - begin

        self.TXF = self.TXF[begin : end]
        self.TXF_close = self.TXF_close[begin : end]
        self.bias_short = self.bias_short[begin : end]
        self.bias_long = self.bias_long[begin : end]
        self.adosc = self.adosc[begin : end]
        self.bband_upper = self.bband_upper[begin : end]
        self.bband_middle = self.bband_middle[begin : end]
        self.bband_lower = self.bband_lower[begin : end]
        self.keltner_upper = self.keltner_upper[begin : end]
        self.keltner_middle = self.keltner_middle[begin : end]
        self.keltner_lower = self.keltner_lower[begin : end]
        self.K = self.K[begin : end]
        self.D = self.D[begin : end]
        self.J = self.J[begin : end]
        self.solid_kbar_diff = self.solid_kbar_diff[begin : end]
        self.upper_shadow = self.upper_shadow[begin : end]
        self.lower_shadow = self.lower_shadow[begin : end]
        self.kbar_diff = self.kbar_diff[begin : end]
        self.cci_short = self.cci_short[begin : end]
        self.cci_long = self.cci_long[begin : end]

    def __len__(self):
        return self.data_len - (SEQ_LEN + MIN_LATER + 1) - 1
    
    def __getitem__(self, idx):
        x_data = torch.cat((
            self.TXF[idx : idx + SEQ_LEN],
            self.bias_short[idx : idx + SEQ_LEN],
            self.bias_long[idx : idx + SEQ_LEN],
            self.adosc[idx : idx + SEQ_LEN],
            self.bband_upper[idx : idx + SEQ_LEN],
            self.bband_middle[idx : idx + SEQ_LEN],
            self.bband_lower[idx : idx + SEQ_LEN],
            self.keltner_upper[idx : idx + SEQ_LEN],
            self.keltner_middle[idx : idx + SEQ_LEN],
            self.keltner_lower[idx : idx + SEQ_LEN],
            self.K[idx : idx + SEQ_LEN],
            self.D[idx : idx + SEQ_LEN],
            self.J[idx : idx + SEQ_LEN],
            self.solid_kbar_diff[idx : idx + SEQ_LEN],
            self.upper_shadow[idx : idx + SEQ_LEN],
            self.lower_shadow[idx : idx + SEQ_LEN],
            self.kbar_diff[idx : idx + SEQ_LEN],
            self.cci_short[idx : idx + SEQ_LEN],
            self.cci_long[idx : idx + SEQ_LEN]
        ), 1)
        last_close_price = self.raw_close_price[idx + SEQ_LEN - 1].item()

        actual_price = self.raw_close_price[idx + SEQ_LEN + MIN_LATER + 1].item()
        price_change_rate = (actual_price - last_close_price) / last_close_price
        y_data = int((price_change_rate + 0.1) / UNIT_PCR)
        return x_data, y_data, actual_price, last_close_price

class PositionEmbedding(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super(PositionEmbedding, self).__init__()
        self.linear = torch.nn.Linear(FEATURE_NUM, d_model, dtype = torch.double)

        # Create a matrix of shape (max_len, feature_num)
        pe = torch.zeros(SEQ_LEN, d_model)
        position = torch.arange(0, SEQ_LEN, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.linear(x)
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        
        # d_model should be divisible by num_heads
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.V_linear = torch.nn.Linear(d_model, d_model, dtype = torch.double)
        self.K_linear = torch.nn.Linear(d_model, d_model, dtype = torch.double)
        self.Q_linear = torch.nn.Linear(d_model, d_model, dtype = torch.double)
        self.output_linear = torch.nn.Linear(d_model, d_model, dtype = torch.double)

    def scaled_dot_product_attention(self, Q: torch.tensor, K: torch.tensor, V: torch.tensor) -> torch.tensor:
        # Q, V: [batch_size, num_heads, seq_len, d_k]
        # K: [batch_size, num_heads, seq_len, d_k] -> K: [batch_size, num_heads, d_k, seq_len]
        # attention: [batch_size, num_heads, seq_len, seq_len]
        attention = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k).float())

        # attention_prob: [batch_size, num_heads, seq_len, seq_len]
        # It satisfies the sum of the last dimension is 1
        # For example, assume the result of this softmax is a
        # then a[0][0][0][0] + a[0][0][0][1] + ... + a[0][0][0][seq_len - 1] = 1
        attention_prob = torch.nn.functional.softmax(attention, dim = -1)
        output = torch.matmul(attention_prob, V)
        return output
    
    def split_heads(self, x: torch.tensor) -> torch.tensor:
        batch_size, seq_len, d_model = x.size()

        # x: [batch_size, seq_len, d_model] -> x: [batch_size, num_heads, seq_len, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        return x
    
    def concat_heads(self, x: torch.tensor) -> torch.tensor:
        batch_size, _, seq_len, d_k = x.size()

        # x: [batch_size, num_heads, seq_len, d_k] -> x: [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return x
    
    def forward(self, Q: torch.tensor, K: torch.tensor, V: torch.tensor) -> torch.tensor:
        Q = self.split_heads(self.Q_linear(Q))
        K = self.split_heads(self.K_linear(K))
        V = self.split_heads(self.V_linear(V))
        # print("-> Split head succeeded!")

        attention_output = self.scaled_dot_product_attention(Q, K, V)
        # print("-> Scaled dot product attention succeeded!")

        output = self.output_linear(self.concat_heads(attention_output))
        # print("-> Concat head succeeded!")
        
        return output

class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super(FeedForward, self).__init__()
        self.fully_connected_1 = torch.nn.Linear(d_model, d_ff, dtype = torch.double)
        self.fully_connected_2 = torch.nn.Linear(d_ff, d_model, dtype = torch.double)
        # self.conv1d_1 = torch.nn.Conv1d(in_channels = d_model, out_channels = d_ff, kernel_size = 1, dtype = torch.double)
        # self.conv1d_2 = torch.nn.Conv1d(in_channels = d_ff, out_channels = d_model, kernel_size = 1, dtype = torch.double)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: [batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
        # x = x.transpose(1, 2)
        # nonliear_output_1 = self.relu(self.conv1d_1(x))
        # output = self.conv1d_2(nonliear_output_1)
        output = self.fully_connected_2(self.relu(self.fully_connected_1(x)))
        # return output.transpose(1, 2)
        return output

class Encoder(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: int) -> None:
        super(Encoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_model * 4)
        self.norm_1 = torch.nn.LayerNorm(d_model, dtype = torch.double)
        self.norm_2 = torch.nn.LayerNorm(d_model, dtype = torch.double)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # print("Start to encode...")
        attention_output = self.multi_head_attention(x, x, x)
        # print("Get the attention output!\n")

        # print("Start to feed forward...")
        feed_forward_input = self.norm_1(x + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(feed_forward_input)
        output = self.norm_2(feed_forward_input + self.dropout(feed_forward_output))
        # print("Get the feed forward output!\n")
        
        return output

class Transformer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout_rate: int) -> None:
        super(Transformer, self).__init__()
        self.d_model = d_model

        self.position_embedding = PositionEmbedding(d_model)
        self.encoder = torch.nn.ModuleList([Encoder(d_model, num_heads, dropout_rate) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.prepare_output = torch.nn.Linear(SEQ_LEN * D_MODEL, 2 * int(0.1 / UNIT_PCR) + 1, dtype = torch.double)
        self.final_layer = torch.nn.LogSoftmax(dim = -1)

    def forward(self, x):
        x = self.position_embedding(x)

        encoder_output = x
        for encoder in self.encoder:
            encoder_output = encoder(encoder_output)

        encoder_output = encoder_output.view(encoder_output.shape[0], -1)

        # [batch_size, seq_len] -> [batch_size, 2*(0.1/UNIT_PCR)+1]
        flatten_encoder_output = self.dropout(self.prepare_output(encoder_output))
        return flatten_encoder_output

        # output = self.final_layer(self.relu(flatten_encoder_output))
        # output = self.final_layer(flatten_encoder_output)
        # return output

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: torch.tensor = None, gamma: int = 2) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.total_class = 2 * int(0.1 / UNIT_PCR) + 1

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:
        ce_loss = torch.nn.CrossEntropyLoss(weight = self.alpha)(input, target).to(device = DEVICE)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss

class early_stopper:
    def __init__(self, patience: int, min_delta_percentage: float) -> None:
        self.patience = patience
        self.min_delta_percentage = min_delta_percentage
        self.counter = 0
        self.best_vloss = None
        self.best_model_path = ""

    def update(self, new_best_vloss: float, model: torch.nn.Module) -> None:
        self.counter = 0
        self.best_vloss = new_best_vloss

        if os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)
        self.best_model_path = f'./models/model_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.pth'
        torch.save(model.state_dict(), self.best_model_path)

    def __call__(self, vloss: float, model: torch.nn.Module) -> bool:
        if self.best_vloss is None:
            self.update(vloss, model)
            return False

        if vloss < self.best_vloss:
            self.update(vloss, model)
            return False
        
        elif vloss > self.best_vloss * (1 + self.min_delta_percentage):
            self.counter += 1
            print(f'Validation loss did not improve for {self.counter} epochs') 
            if self.counter >= self.patience:
                return True
            return False
# ================================================== #


# ========== Functions ========== #
def draw_plot(x_data: pd.DataFrame, y_data: pd.DataFrame, x_label: str, y_label: str) -> None:
    fig = plt.figure(figsize=(7, 4))
    plt.scatter(x_data, y_data, s = 1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def keltner_bands(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int, multiplier: int, position: int) -> tuple:
    mid = ta.EMA(close, timeperiod = period)
    mid = np.nan_to_num(mid, nan = mid.iloc[period - 1])
    kelt_trange = np.array([])

    for i in tqdm(range(1, len(close)), desc = "Calculating kelner bands ", position = position, leave = TQDM_LEAVE):
        tem_trange = max(
            high.iloc[-i] - low.iloc[-i],
            abs(high.iloc[-i] - close.iloc[-i - 1]),
            abs(low.iloc[-i] - close.iloc[-i - 1])
        )
        kelt_trange = np.append(tem_trange, kelt_trange)
    kelt_trange = np.append(high.iloc[0] - low.iloc[0], kelt_trange)
    atr = ta.EMA(kelt_trange, timeperiod = period)
    atr = np.nan_to_num(atr, nan = atr[period - 1])
    upper = mid + atr * multiplier
    lower = mid - atr * multiplier

    return upper, mid, lower

def KDJ(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, signal_k: int, signal_d: int, position: int) -> tuple:
    _alpha_k = 2 / (signal_k + 1)
    _alpha_d = 2 / (signal_d + 1)

    lowest = ta.MIN(low, timeperiod = period)
    lowest = np.nan_to_num(lowest, nan = lowest.iloc[period - 1])
    highest = ta.MAX(high, timeperiod = period)
    highest = np.nan_to_num(highest, nan = highest.iloc[period - 1])

    rsv = (close - lowest) / (highest - lowest) * 100
    
    K = np.array([50])
    D = np.array([50])
    J = np.array([50])
    
    for i in tqdm(range(1, len(close)), desc = "Calculating KDJ ", position = position, leave = TQDM_LEAVE):
        K = np.append(K, int(_alpha_k * ((K[-1] + 2 * rsv.iloc[i]) / 3) + (1 - _alpha_k) * K[-1] + 0.5))
        D = np.append(D,  int(_alpha_d * ((D[-1] + 2 * K[-1]) / 3) + (1 - _alpha_d) * D[-1] + 0.5))
        J = np.append(J, 3 * K[-1] - 2 * D[-1])

    return K, D, J

def load_and_check(data_path: str, start_date: datetime.date = None, end_date: datetime.date = None) -> pd.DataFrame:
    global BEGIN_TIME, END_TIME

    data = pd.read_csv(data_path, dtype = {
        'Date': str,
        'open': np.int16,
        'high': np.int16,
        'low': np.int16,
        'close': np.int16,
        'volume': np.int16
    }, index_col = 0)
    data.index = pd.to_datetime(data.Date)
    data = data.between_time(BEGIN_TIME, END_TIME)

    if start_date is not None:
        data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]
 
    IS_MISSING_DATA = False
    missing_time_index = []
    for i in range(1, len(data.index)):
        if data.index[i] - data.index[i - 1] != datetime.timedelta(minutes = 1) and data.index[i].time() != BEGIN_TIME:
            if IS_MISSING_DATA == False:
                IS_MISSING_DATA = True
                print('Not continuous time: ')
                
            print('    ', data.index[i - 1], data.index[i])
            missing_time_index.append(i - 1)

    finish = 0
    if IS_MISSING_DATA:
        print('=' * 50)

        for int_index in missing_time_index:
            time_delta = (data.index[int_index + finish + 1] - data.index[int_index + finish]).seconds // 60
            Entity_delta = data.Open.iloc[int_index + finish + 1] - data.Close.iloc[int_index + finish]
            High_delta = data.High.iloc[int_index + finish + 1] - data.High.iloc[int_index + finish]
            Low_delta = data.Low.iloc[int_index + finish + 1] - data.Low.iloc[int_index + finish]
            Volume_delta = data.Volume.iloc[int_index + finish + 1] - data.Volume.iloc[int_index + finish]
            print(Entity_delta)

            for minute in range(1, time_delta):
                print(f"Missing data at {data.index[int_index + finish]}")
                print(data.iloc[int_index + finish - 1: int_index + finish + 4][:-1], end = '\n\n')
                
                time_for_missing_data = (data.index[int_index + finish] + datetime.timedelta(minutes = 1))
                new_data = pd.DataFrame({
                    'Date': time_for_missing_data,
                    'Open': data.Close.iloc[int_index + finish],
                    'High': round(data.High.iloc[int_index + finish - minute + 1] + High_delta * minute / (time_delta - 1)),
                    'Low': round(data.Low.iloc[int_index + finish - minute + 1] + Low_delta * minute / (time_delta - 1)),
                    'Close': round(data.Close.iloc[int_index + finish - minute + 1] + Entity_delta * minute / (time_delta - 1)),
                    'Volume': round(data.Volume.iloc[int_index + finish - minute + 1] + Volume_delta * minute / (time_delta - 1))
                }, index = [time_for_missing_data])
                data = pd.concat([data.iloc[:int_index + finish + 1], new_data, data.iloc[int_index + finish + 1:]])
                finish += 1
                print(data.iloc[int_index + finish - 1: int_index + finish + 4], end = '\n\n')

        int_index_for_all_data = [i for i in range(len(data))]
        data.insert(0, '', int_index_for_all_data)
        data.to_csv(data_path, index = False)

        load_and_check(data_path)
    else:
        print(f"Succeed to load and check the data at '{data_path}'")
        return data
    
def init_model(loss_weight = None):
    transformer = Transformer(
        d_model = D_MODEL,
        num_heads = NUM_HEADS,
        num_layers = NUM_ENCODER_LAYERS,
        dropout_rate = DROPOUT_RATE
    ).to(device = DEVICE)

    criterion = FocalLoss(alpha = loss_weight, gamma = 2)
    optimizer = torch.optim.Adam(transformer.parameters(), lr = LEARNING_RATE, betas=(0.9, 0.98), eps=1e-8, weight_decay = 0.02)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 1e-5, max_lr = 1e-3)

    return transformer, criterion, optimizer, lr_scheduler

def train_model(epoch: int, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler, device: str) -> None:
    logging.basicConfig(filename = LOG_FILE_PATH, filemode = 'a', format = '%(asctime)s [%(levelname)s] %(message)s', level = logging.INFO)

    model.to(device = device)
    model.train()

    tloss = 0.0
    first_batch = True
    for x_data, y_data, _, _ in tqdm(dataloader, desc = f'Epoch {epoch + 1}: training...', position = 1, leave = TQDM_LEAVE):
        x_data = x_data.to(device = device)
        y_data = y_data.to(device = device)
        y_data = y_data.to(torch.int64)

        if torch.all(x_data[0] == x_data[-1]):
            raise ValueError('Something wrong with the data')
        
        if first_batch:
            logging.info('-' * 50)
            logging.info(f'In the first batch of epoch {epoch + 1}:')
            logging.info(f'    x_data[0][0]: {["{:.5f}".format(num) for num in x_data[0][0].tolist()]}')
            logging.info(f'    x_data[0][1]: {["{:.5f}".format(num) for num in x_data[0][1].tolist()]}')
            logging.info(f'    y_data[0]: {y_data[0]}')
            logging.info(f'    y_data[1]: {y_data[1]}')

        output = model(x_data)
        loss = criterion(output, y_data)

        if first_batch:
            logging.info(f'    output[0]: {["{:.5f}".format(num) for num in output[0].tolist()]}')
            logging.info(f'    output[1]: {["{:.5f}".format(num) for num in output[1].tolist()]}')
            logging.info(f'    loss: {loss}')
            first_batch = False

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        tloss += loss.item()

    lr_scheduler.step()

    average_tloss = tloss / len(dataloader.dataset)
    logging.info(f'Epoch: {epoch + 1}, Total data: {len(dataloader.dataset)}')
    logging.info(f'Epoch: {epoch + 1}, Average Loss: {average_tloss}')
    logging.info('-' * 50)

    return tloss
    
def validate_model(epoch: int, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str) -> float:
    logging.basicConfig(filename = LOG_FILE_PATH, filemode = 'a', format = '%(asctime)s [%(levelname)s] %(message)s', level = logging.INFO)

    model.to(device = device)
    model.eval()

    vloss = 0.0
    error = 0.0
    max_error = 0.0
    class_error = [0] * (2 * int(0.1 / UNIT_PCR) + 1)
    first_batch = True
    with torch.no_grad():
        for x_data, y_data, actual_price, last_close_price in tqdm(dataloader, desc = f'Epoch {epoch + 1}: validating...', position = 0, leave = TQDM_LEAVE):
            x_data = x_data.to(device = device)  # x_data = [BATCH_SIZE, SEQ_LEN, FEATURE_NUM]
            y_data = y_data.to(device = device)  # y_data = [BATCH_SIZE]
            y_data = y_data.to(torch.int64)

            output = model(x_data)
            loss = criterion(output, y_data)
            vloss += loss.item()

            class_number = torch.argmax(output, dim = -1)
            class_err = class_number - y_data
            for i in range(len(class_err)):
                class_error[class_err[i] + int(0.1 / UNIT_PCR)] += 1

            price_change_rate = torch.round((class_number - int(0.1 / UNIT_PCR)) * UNIT_PCR, decimals = 3)
            predict_price = last_close_price * (1 + price_change_rate)
            error += torch.sum(torch.abs(predict_price - actual_price))
            max_error = max(max_error, torch.max(torch.abs(predict_price - actual_price)))

            if first_batch:
                logging.info('-' * 50)
                logging.info(f'In the first batch of epoch {epoch + 1}:')
                logging.info(f'    output[0]: {["{:.5f}".format(num) for num in output[0].tolist()]}')
                logging.info(f'    y_data: {[num for num in y_data.tolist()]}')
                logging.info(f'    class_number: {[num for num in class_number.tolist()]}')
                logging.info(f'    price_change_rate: {[num for num in price_change_rate.tolist()]}')
                logging.info(f'    original_price: {["{:.5f}".format(num) for num in last_close_price.tolist()]}')
                logging.info(f'    predict_price: {["{:.5f}".format(num) for num in predict_price.tolist()]}')
                logging.info(f'    actual_price: {["{:.5f}".format(num) for num in actual_price.tolist()]}')
                first_batch = False

    average_vloss = vloss / len(dataloader.dataset)
    average_error = error / len(dataloader.dataset)
    accuracy = class_error[int(0.1 / UNIT_PCR)] / len(dataloader.dataset)

    logging.info(f'Epoch: {epoch + 1}, Total data: {len(dataloader.dataset)}')
    logging.info(f'Epoch: {epoch + 1}, Average Validation Loss: {average_vloss}')
    logging.info(f'Epoch: {epoch + 1}, Average Error: {average_error}')
    logging.info(f'Epoch: {epoch + 1}, Max Error: {max_error}')
    logging.info(f'Epoch: {epoch + 1}, Class Error: {class_error}')
    logging.info(f'Epoch: {epoch + 1}, Accuracy: {accuracy}')
    logging.info('-' * 50)
    
    return vloss, average_error, max_error, accuracy
# ================================================== #

if __name__ == '__main__':
    torch.set_printoptions(linewidth = 300)
    with open(LOG_FILE_PATH, 'w') as f:
        f.write('')
    logging.basicConfig(filename = LOG_FILE_PATH, filemode = 'a', format = '%(asctime)s [%(levelname)s] %(message)s', level = logging.INFO)

    if WANDB_RELATED:
        logging.info('Initiating wandb...')
        wandb.init(
            project = "transformer",

            config = {
                "architecture": "transformer",
                "dataset": "TXF",
                "Kbar_timeunit": "1min",
                "LOSS_FUNC": "CrossEntropyLoss",
                "OPTIMIZER": "adam",

                "SEQ_LEN": SEQ_LEN,
                "BATCH_SIZE": BATCH_SIZE,
                "EPOCHS": EPOCHS,
                "D_MODEL": D_MODEL,
                "NUM_HEADS": NUM_HEADS,
                "NUM_ENCODER_LAYERS": NUM_ENCODER_LAYERS,
                "DROPOUT_RATE": DROPOUT_RATE,
                "LEANING_RATE": LEARNING_RATE,
                "KFold": K,
                "PATIENCE": PATIENCE,
                "MIN_DELTA_PERCENTAGE": MIN_DELTA_PERCENTAGE,
                "UNIT_PCR": UNIT_PCR,

                "MIN_LATER": MIN_LATER,

                "EPSILON": 1e-8,
                "BETA": (0.9, 0.98),
                "WEIGHT_DECAY": 0.02,
            }
        )
        logging.info('Succeed to initiate wandb!')
        logging.info('=' * 50)

    logging.info("Loading data...")
    TXF = load_and_check('data/TXF_long.csv')
    TXF_test = load_and_check('data/TXFR1_1min.csv', start_date = datetime.date(2023, 1, 1), end_date = datetime.date(2024, 8, 16))
    logging.info(f'TXF shape: {TXF.shape}')
    logging.info(f'TXF_test shape: {TXF_test.shape}')

    ## Split the data into training, validating and testing set
    TXF_train_and_valid = TXF[(TXF.index.date < datetime.date(2023, 1, 1))]
    TXF_test = pd.concat([TXF[(TXF.index.date >= datetime.date(2023, 1, 1))], TXF_test[(TXF_test.index.date >= datetime.date(2023, 12, 9))]])
    logging.info(f'train and validate size: {TXF_train_and_valid.shape[0]}')
    logging.info(f'test size: {TXF_test.shape[0]}')
    # draw_plot(TXF_train.index, TXF_train.Close, 'TXF Close', 'Time')
    logging.info("Succeed to load and check the data!")
    logging.info('=' * 50)

    logging.info("Creating dataset and dataloader...")
    with ProcessPoolExecutor() as executor:
        train_valid_dataset_future = executor.submit(custom_dataset, 0, TXF_train_and_valid.shape[0])
        test_dataset_future = executor.submit(custom_dataset, TXF_train_and_valid.shape[0], TXF_train_and_valid.shape[0] + TXF_test.shape[0])

        train_valid_dataset = train_valid_dataset_future.result()
        test_dataset = test_dataset_future.result()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True, drop_last = True)
    test = iter(torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False))
    test_x, test_y, _, _ = next(test)
    FEATURE_NUM = test_x.shape[2]

    logging.info(f'In every data: ')
    logging.info(f'    x_data shape: {test_x[0].shape}')
    logging.info(f'    y_data shape: {test_y[0].shape}')
    logging.info(f'Which means we use the past [{SEQ_LEN}] minutes data to predict the price in [{MIN_LATER}] minutes later')
    logging.info(f'Succeed to create dataset and dataloader!')
    logging.info('=' * 50)

    os.system('cls')

    torch.cuda.device(0)
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)

    class_distribution = torch.load('./tensor/class_distribution.pt', weights_only = True)
    # loss_weight = ((torch.sum(class_distribution) / class_distribution) ** 2).to(device = DEVICE)
    transformer, criterion, optimizer, lr_scheduler = init_model()

    kfold = KFold(n_splits = K, shuffle = True)
    for fold, (train_index, valid_index) in enumerate(kfold.split(train_valid_dataset)):
        logging.info(f'Fold: {fold + 1}')
        
        train_subsampler = torch.utils.data.Subset(train_valid_dataset, train_index)
        valid_subsampler = torch.utils.data.Subset(train_valid_dataset, valid_index)

        train_loader = torch.utils.data.DataLoader(train_subsampler, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True, drop_last = True)
        valid_loader = torch.utils.data.DataLoader(valid_subsampler, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True, drop_last = True)

        transformer, criterion, optimizer, lr_scheduler = init_model()
        early_stop = early_stopper(patience = PATIENCE, min_delta_percentage = MIN_DELTA_PERCENTAGE)

        cur_model = None
        vloss_future = None
        with ProcessPoolExecutor() as executor:
            # training loop
            for epoch in range(EPOCHS):
                train = executor.submit(train_model, epoch, transformer, train_loader, criterion, optimizer, lr_scheduler, DEVICE)
                if cur_model is not None:
                    vloss_future = executor.submit(validate_model, epoch - 1, cur_model, valid_loader, criterion, 'cpu')

                if vloss_future is not None:
                    vloss, average_error, max_error, accuracy = vloss_future.result()
                    average_vloss = vloss / len(valid_loader)
                    if WANDB_RELATED:
                        wandb.log({'validation_loss': vloss})
                        wandb.log({'average_validation_loss': average_vloss})
                        wandb.log({'average_error': average_error})
                        wandb.log({'max_error': max_error})
                        wandb.log({'accuracy': accuracy})

                    if early_stop(average_error, cur_model):
                        break

                tloss = train.result()
                average_tloss = tloss / len(train_loader)
                if WANDB_RELATED:
                    wandb.log({'loss': tloss})
                    wandb.log({'average_loss': average_tloss})   

                cur_model = copy.deepcopy(transformer)

        torch.cuda.empty_cache()
        logging.info('=' * 50)