#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/14 

# to understand what NIQE refers...

import random
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

from run_utils import *

IMG_FILE = IN_PATH / '0001.png'
PATCH_SIZE = 100

img = Image.open(IMG_FILE)
w, h = img.size
x = random.randrange(w - PATCH_SIZE)
y = random.randrange(h - PATCH_SIZE)
patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
print('patch.size', patch.size)
im = np.asarray(patch, dtype=np.float32) / 255.0
print('im.shape', im.shape)
im_bgr = im[:, :, ::-1]   # RGB2BGR
im_y = bgr2ycbcr(im_bgr, y_only=True)     # [H, W], RGB => Y
im_y: ndarray = np.round(im_y * 255)      # float32 =? uint8
print('im_y.shape', im_y.shape)


niqe_scores = []

def func(x:ndarray) -> float:
  im_x = np.asarray(x, dtype=np.float32).reshape(im_y.shape)
  niqe_score = niqe(im_x, mu_pris_param, cov_pris_param, gaussian_window)
  niqe_scores.append(niqe_score)
  return niqe_score


# PGD settings
steps = 10
eps   = 8
alpha = 1

xk = im_y.flatten()
for _ in tqdm(range(steps)):
  grad = approx_fprime(xk, func, epsilon=eps)
  xk -= np.sign(grad) * alpha
  xk = np.clip(xk, 0, 255)

im_y_hat = np.asarray(xk).reshape(im_y.shape)

plt.clf()
plt.subplot(221) ; plt.title('y_ch')     ; plt.imshow(im_y)
plt.subplot(222) ; plt.title('y_ch opt') ; plt.imshow(im_y_hat)
plt.subplot(223) ; plt.title('rgb')      ; plt.imshow(im)
plt.subplot(224) ; plt.title('niqe')     ; plt.plot(niqe_scores)
plt.tight_layout()
fp = OUT_PATH / 'attack_niqe.png'
print(f'>> save to {fp}')
plt.savefig(fp, dpi=600)
