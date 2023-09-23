import torch
import numpy as np
import os
import torch.nn as nn
import scipy.io
import h5py
import pickle
from scipy import interpolate
import hdf5storage as hdf5


# n_f = 256
# n_c = 64
# N = 1000
# T = 15
# x_f, y_f = np.linspace(0, 1, n_f), np.linspace(0, 1, n_f)
# x_c, y_c = np.linspace(0, 1, n_c), np.linspace(0, 1, n_c)
# x_interp = []
#
# a = hdf5.loadmat('/data/ns_1e-6_T15_val.mat')['a']
# x = hdf5.loadmat('/data/ns_1e-6_T15_val.mat')['u']
#
# for i in range(N):
#     x_interp_T = []
#     for j in range(T):
#         xi_interp = interpolate.interp2d(x_f, y_f, x[i,:,:,j])
#         x_interp_T.append(xi_interp(x_c, y_c))
#     x_interp.append(np.stack(x_interp_T, axis=-1))
# u = np.stack(x_interp, axis=0)

# hdf5.savemat('/NS64_1e-6_T15_val.mat', mdict={'a': a, 'u': u}, format='7.3', matlab_compatible = True)


device = torch.device('cuda')
n_f = 256
n_c = 64
N = 4
T = 3
x_f, y_f = np.linspace(0, 1, n_f), np.linspace(0, 1, n_f)
x_c, y_c = np.linspace(0, 1, n_c), np.linspace(0, 1, n_c)
x_interp = []

x = torch.randn((4,256,256,3),device=device)

for i in range(N):
    x_interp_T = []
    for j in range(T):
        xi_interp = interpolate.interp2d(x_f, y_f, x[i,:,:,j].cpu().numpy())
        x_interp_T.append(xi_interp(x_c, y_c))
    x_interp.append(np.stack(x_interp_T, axis=-1))
u = np.stack(x_interp, axis=0)