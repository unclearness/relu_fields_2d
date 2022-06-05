import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import torch.nn.functional as F

device = torch.device('cuda:0')


def convertImage(torch_img):
    numpy_img = torch_img.permute(
        3, 2, 1, 0).squeeze(-1).to('cpu').detach().numpy().copy()
    numpy_img = (np.clip(numpy_img, 0, 1) *
                 255).astype(np.uint8).transpose(1, 0, 2)
    return numpy_img


def sampleImage(grid_param, grid, activation=None):
    sampled = F.grid_sample(grid_param, grid, mode='bilinear',
                            padding_mode='zeros',
                            align_corners=True)
    if activation is not None:
        sampled = activation(sampled)
    sampled = torch.clamp(sampled, min=0.0, max=1.0)
    return sampled


def drawGrid(img, grid_row_num, grid_col_num, color=(0, 0, 255)):
    if len(img.shape) == 2:
        img = np.repeat(img.unsqueeze(-1), 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)
    h, w, c = img.shape
    for j in range(grid_row_num):
        r = int(j * h / grid_row_num)
        img = cv2.line(img, (0, r), (w - 1, r), color)
    for i in range(grid_col_num):
        c = int(i * w / grid_col_num)
        img = cv2.line(img, (c, 0), (c, h - 1), color)
    return img


def sampleReleuImage(grid_param, grid):
    zero = torch.zeros((1), device=device)
    def f(x): return torch.max(x, zero)
    return sampleImage(grid_param, grid, activation=f)


if __name__ == '__main__':

    out_dir = "./out/"
    os.makedirs(out_dir, exist_ok=True)

    dst_path = "./data/train.png"
    dst = cv2.imread(dst_path, -1)
    if len(dst.shape) == 2:
        h, w = dst.shape
        c = 1
    else:
        h, w, c = dst.shape
        c = 1
        dst = dst[..., 0]
    dst = torch.from_numpy(dst.reshape(
        1, c, h, w).astype(np.float32) / 255).to(device)
    pix_num = h * w * c
    # print(dst.shape)

    dst_pix_per_row = 2
    dst_pix_per_col = 12

    grid_row_num = (int(h / dst_pix_per_row) + 1)
    grid_col_num = (int(w / dst_pix_per_col) + 1)

    grid = np.repeat(
        np.arange(pix_num).reshape((c, h, w, 1)), 2, axis=-1)
    # pix scale indice
    grid[..., 0] = grid[..., 0] / w
    grid[..., 1] = grid[..., 1] % w

    # -1~1 scale indice
    grid = grid.astype(np.float32)
    grid[..., 0] = (grid[..., 0] / h) * 2 - 1
    grid[..., 1] = (grid[..., 1] / w) * 2 - 1

    grid = torch.from_numpy(grid).to(device)

    setup = [('relue', sampleReleuImage), ('naive', sampleImage)]
    for alg_name, sample_f in setup:
        alg_dir = out_dir + "/" + alg_name + "/"
        os.makedirs(alg_dir, exist_ok=True)

        grid_param = torch.rand(
            (1, c, grid_row_num, grid_col_num), device=device)
        grid_param = nn.Parameter(grid_param)

        max_iter = 1000
        optimizer = torch.optim.Adam(
            [grid_param], lr=0.05)
        for i in range(max_iter):
            optimizer.zero_grad()

            sampled = sample_f(grid_param, grid)
            diff = sampled - dst
            loss = (diff * diff).sum()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(i, loss)
            img = convertImage(sampled)
            # img = drawGrid(img, grid_row_num, grid_col_num)
            cv2.imwrite(f"{alg_dir}/{i}.png", img)
        print(i, loss, sampled)
