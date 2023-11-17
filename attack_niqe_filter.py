#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/14 

# to understand what NIQE refers...

from PIL import ImageFilter

from run_utils import *

IMG_FILE = IN_PATH / '0001.png'
PATCH_SIZE = 100

pil_to_np = lambda img: np.asarray(img, dtype=np.float32) / 255.0


img = Image.open(IMG_FILE)
print('original:', get_niqe(pil_to_np(img)))

print('blur:',     get_niqe(pil_to_np(img.filter(ImageFilter.BLUR))))
print('smooth:',   get_niqe(pil_to_np(img.filter(ImageFilter.SMOOTH))))
print('smooth++:', get_niqe(pil_to_np(img.filter(ImageFilter.SMOOTH_MORE))))
print('gauss(2):', get_niqe(pil_to_np(img.filter(ImageFilter.GaussianBlur(2)))))
print('gauss(4):', get_niqe(pil_to_np(img.filter(ImageFilter.GaussianBlur(4)))))

print('sharpen:',  get_niqe(pil_to_np(img.filter(ImageFilter.SHARPEN))))
print('detail:',   get_niqe(pil_to_np(img.filter(ImageFilter.DETAIL))))
print('edge:',     get_niqe(pil_to_np(img.filter(ImageFilter.EDGE_ENHANCE))))
print('edge++:',   get_niqe(pil_to_np(img.filter(ImageFilter.EDGE_ENHANCE_MORE))))

print('contour:',  get_niqe(pil_to_np(img.filter(ImageFilter.CONTOUR))))
print('findegde:', get_niqe(pil_to_np(img.filter(ImageFilter.FIND_EDGES))))
print('emboss:',   get_niqe(pil_to_np(img.filter(ImageFilter.EMBOSS))))
print('usmsk(2):', get_niqe(pil_to_np(img.filter(ImageFilter.UnsharpMask(2)))))
print('usmsk(4):', get_niqe(pil_to_np(img.filter(ImageFilter.UnsharpMask(4)))))
