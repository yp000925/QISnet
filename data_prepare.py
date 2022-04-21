import os
import numpy as np

pth=  'VOC2012/JPEGImages'
pth_train='VOC2012/train_data'
pth_test='VOC2012/test_data'
pth_val= 'VOC2012/val_data'


filenames = [f for f in os.listdir(pth) if f.endswith(".jpg") or f.endswith(".png")]

filenames.sort()
filenames = filenames[0:10000]
l = len(filenames)
t1 = round(l*0.7)
t2 = round(l*0.2)

t3 = round(l*0.1)

train_file = filenames[0:t1]
test_file = filenames[t1:(t1+t2)]
val_file = filenames[(t1+t2)::]

for name in train_file:
    try:
        src = os.path.join(pth, name)
        dst = os.path.join(pth_train,name)
        os.rename(src,dst)
    except:
        print(name)
        break

for name in test_file:
    try:
        src = os.path.join(pth, name)
        dst = os.path.join(pth_test,name)
        os.rename(src,dst)
    except:
        print(name)
        break

for name in val_file:
    try:
        src = os.path.join(pth, name)
        dst = os.path.join(pth_val,name)
        os.rename(src,dst)
    except:
        print(name)
        break


#%%
from PIL import Image
train_file = [f for f in os.listdir(pth_val) if f.endswith(".jpg") or f.endswith(".png")]
patch_sz = 128
jit = 0
J = 2
window_sz = (patch_sz+2*jit)*J
abandon = []
cnt = 0
for name in train_file:
    img_pth = os.path.join(pth_val,name)
    image = np.array(Image.open(img_pth).convert('L')) / 255.0
    [height, width] = image.shape

    if height < window_sz or width < window_sz:
        abandon.append(name)
        cnt +=1

#%%
for name in abandon:
    try:
        src = os.path.join(pth_val,name)
        dst = os.path.join(pth,name)
        os.rename(src,dst)
    except:
        print(name)
        break
