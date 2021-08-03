import CNN as cn
import CNN_LeNet
import CNN_AlexNet
import GoogleNet as gn
import VGG as v
import time
import torch
from torch import nn, optim
import d2lzh_pytorch as d2l
import NIN as nin
import Normalization as nor
import ResNet as rn
import DenseNet as dn
import RNN as rnn
import GRU as gru
import LSTM as lstm
import random
import zipfile

with zipfile.ZipFile("data/jaychou_lyrics.txt.zip") as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
    # print(corpus_chars)
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]
# print(corpus_chars)

idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print(vocab_size)
# corpus 资料库
corpus_indices = [char_to_idx[char] for char in corpus_chars]
num_epochs, num_steps, batch_size, lr, cliping_theta = 160, 35, 32, 1e-2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
params = lstm.get_params(num_inputs, num_hiddens, num_outputs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rnntest1():
    with zipfile.ZipFile("data/jaychou_lyrics.txt.zip") as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    # print(corpus_chars)
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    # print(corpus_chars)

    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    print(vocab_size)
    # corpus 资料库
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    # sample = corpus_indices[:20]
    #
    # print(sample)
    #
    # my_seq = list(range(30))
    # for X, Y in d2l.data_iter_random(my_seq, batch_size=2, num_steps=6):
    #     print('X: ', X, '\nY:', Y, '\n')

    X = torch.arange(10).view(2, 5)
    inputs = d2l.to_onehot(X, vocab_size)
    print(len(inputs), inputs[0].shape)

    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 简单测试来观察结果的个数（时间步数）
    # state = rnn.init_rnn_state(X.shape[0], num_hiddens)
    # inputs = d2l.to_onehot(X.to(device), vocab_size)
    params = rnn.get_params(num_inputs, num_hiddens, num_outputs)
    # outputs, state_new = rnn.rnn(inputs, state, params)
    # print(len(outputs), outputs[0].shape, state_new[0].shape)

    print(d2l.predict_rnn('分开', 10, rnn.rnn, params, rnn.init_rnn_state, num_hiddens, vocab_size, device, idx_to_char,
                          char_to_idx))

    num_epochs, num_steps, batch_size, lr, cliping_theta = 250, 35, 32, 1e2, 1e-2

    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    d2l.train_and_predict_rnn(rnn.rnn, params, rnn.init_rnn_state, num_hiddens, vocab_size, device, corpus_indices,
                              idx_to_char, char_to_idx, False, num_epochs, num_steps, lr, cliping_theta, batch_size,
                              pred_period, pred_len, prefixes)


def rnntest2():
    num_hiddens = 256
    # rnn_layer=nn.LSTM(input_size=vocab_size,hidden_size=num_hiddens) # 已测试
    rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

    num_steps = 35
    batch_size = 2
    state = None
    X = torch.rand(num_steps, batch_size, vocab_size)
    Y, state_new = rnn_layer(X, state)
    print(Y.shape, len(state_new), state_new[0].shape)

    num_epochs, batch_size, lr, cliping_theta = 250, 32, 1e-3, 1e-2  # 注意这里的学习率设置
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = d2l.RNNModel(rnn_layer, vocab_size).to(device)
    d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                                      num_epochs, num_steps, lr, cliping_theta, batch_size, pred_period, pred_len,
                                      prefixes)


def grutest3():
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = gru.get_params(num_inputs, num_hiddens, num_outputs)
    d2l.train_and_predict_rnn(gru.gru, params, gru.init_gru_state,
                              num_hiddens, vocab_size, device,
                              corpus_indices,
                              idx_to_char, char_to_idx, False,
                              num_epochs, num_steps, lr, cliping_theta, batch_size, pred_period, pred_len,
                              prefixes)

    d2l.train_and_predict_rnn_pytorch()


def grutest4():
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = gru.get_params(num_inputs, num_hiddens, num_outputs)

    gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
    model = d2l.RNNModel(gru_layer, vocab_size).to(device)
    d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                                      num_epochs, num_steps, lr, cliping_theta, batch_size, pred_period, pred_len,
                                      prefixes)

def LSTMtest():
    # d2l.train_and_predict_rnn(lstm.lstm, params, lstm.init_lstm_state, num_hiddens, vocab_size, device, corpus_indices,
    #                           idx_to_char, char_to_idx, False, num_epochs, num_steps, lr, cliping_theta, batch_size,
    #                           pred_period, pred_len, prefixes)

    lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
    model = d2l.RNNModel(lstm_layer, vocab_size).to(device)
    d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                                      num_epochs, num_steps, lr, cliping_theta, batch_size, pred_period, pred_len,
                                      prefixes)
