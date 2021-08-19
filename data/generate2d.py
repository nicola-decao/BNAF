import torch
import sklearn
import numpy as np


def sample2d(data, batch_size=200):

    rng = np.random.RandomState()

    if data == "8gaussians":
        scale = 4.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    else:
        raise RuntimeError


def energy2d(data, z):

    if data == "t1":
        return U1(z)
    elif data == "t2":
        return U2(z)
    elif data == "t3":
        return U3(z)
    elif data == "t4":
        return U4(z)
    else:
        raise RuntimeError


def w1(z):
    return torch.sin(2 * np.pi * z[:, 0] / 4)


def w2(z):
    return 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)


def w3(z):
    return 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)


def U1(z):
    z_norm = torch.norm(z, 2, 1)
    add1 = 0.5 * ((z_norm - 2) / 0.4) ** 2
    add2 = -torch.log(
        torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
        + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
        + 1e-9
    )

    return add1 + add2


def U2(z):
    return 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2


def U3(z):
    in1 = torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
    in2 = torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
    return -torch.log(in1 + in2 + 1e-9)


def U4(z):
    in1 = torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    in2 = torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    return -torch.log(in1 + in2 + 1e-9)
