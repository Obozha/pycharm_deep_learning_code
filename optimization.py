import torch
import numpy as np
import d2lzh_pytorch as d2l
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x
        results.append(x)
    print('epoch 10,x:', x)
    return results


def f(x):
    return x * np.cos(np.pi * x)


def localmin():
    d2l.set_figsize((4.5, 2.4))
    x = np.arange(-1.0, 2.0, 0.1)
    fig, = d2l.plt.plot(x, f(x))
    fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0), arrowprops=dict(arrowstyle='->'))
    fig.axes.annotate('local maximum', xy=(1.1, -0.95), xytext=(0.6, 0.8), arrowprops=dict(arrowstyle='->'))
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')
    d2l.plt.show()


def saddlepoint():
    x = np.arange(-2.0, 2.0, 0.1)
    fig = d2l.plt.plot(x, x ** 3)
    fig[0].axes.annotate('saddle point', xy=(0, 0), xytext=(-0.40, -5.0), arrowprops=dict(arrowstyle='->'))
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')
    d2l.plt.show()


def saddlepoint3d():
    x, y = np.mgrid[-1:1:31j, -1:1:31j]
    z = x ** 2 - y ** 2
    ax = d2l.plt.figure().add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
    ax.plot([0], [0], [0], 'rx')
    ticks = [-1, 0, 1]
    d2l.plt.xticks(ticks)
    d2l.plt.yticks(ticks)
    ax.set_zticks(ticks)
    d2l.plt.xlabel('x')
    d2l.plt.xlabel('y')
    d2l.plt.show()


def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    d2l.set_figsize()
    d2l.plt.plot(f_line, [x * x for x in f_line])  # 图像
    d2l.plt.plot(res, [x * x for x in res], '-o')  # 梯度下降的值
    d2l.plt.xlabel('x')
    d2l.plt.xlabel('f(x)')
    d2l.plt.show()


def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f,x2 %f' % (i + 1, x1, x2))
    return results


def show_trace_2d(f, results):
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), color='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.xlabel('x2')
    d2l.plt.show()


def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2, eta=0.1):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)


def sgd_2d(x1, x2, s1, s2, eta=0.1):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)), x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)


def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2, eta=0.6):
    return x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0


def momentum_2d(x1, x2, v1, v2, gamma=0.5, eta=0.6):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2


def init_momentum_states(features, labels):
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data


def adagrad_2d(x1, x2, s1, s2, eta=2):
    # 前两项为自变量梯度
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6

    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def init_adagrad_states(features):
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)


def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s.data += (p.grad.data ** 2)
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)


def rmsprop_2d(x1, x2, s1, s2, eta=0.4, gamma=0.9):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2


def init_rmsprop_states(features):
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)


def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s.data = gamma * s.data + (1 - gamma) * p.grad.data ** 2
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)


def init_adadelta_states(features):
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    delta_w, delta_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((s_w, delta_w), (s_b, delta_b))


def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * (p.grad.data ** 2)
        g = p.grad.data * torch.sqrt((delta + eps) / (s + eps))
        p.data -= g
        delta[:] = rho * delta + (1 - rho) * g * g


def init_adam_states(features):
    v_w, v_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((v_w, s_w), (v_b, s_b))


def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad.data
        s[:] = beta2 * s + (1 - beta2) * p.grad.data ** 2
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p.data -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
