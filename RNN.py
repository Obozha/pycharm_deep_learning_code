import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_params(num_inputs, num_hiddens, num_outputs):
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return torch.nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def init_rnn_state(batch_size, num_hiddens, device=device):
    return torch.zeros((batch_size, num_hiddens), device=device)


def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为（batch_size, vocab_size）的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, H
