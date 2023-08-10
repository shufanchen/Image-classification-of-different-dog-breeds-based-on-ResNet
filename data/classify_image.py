import pandas as pd
import os
import shutil

labels = pd.read_csv('labels.csv')
labels_group = labels.groupby('breed')
types = pd.DataFrame(labels_group.dtypes).index
rate = 0.1
path_train = 'train_'+str(rate)
path_src = r'train'
path_val = 'val_'+str(rate)

for i in range(120):
    group_temp = labels_group.get_group(types[i])
    spilt_num = rate*int(len(group_temp))  #rate is the rate of validation set
    path_train_dst = path_train+'/'+types[i]
    path_val_dst = path_val+'/'+types[i]
    os.makedirs(path_train_dst, exist_ok=True)
    os.makedirs(path_val_dst, exist_ok=True)
    for j in range(len(group_temp)):
        if j>spilt_num:
            image_src = path_src+'/'+group_temp['id'].iloc[j]+'.jpg'
            image_dst = path_train_dst+'/'+group_temp['id'].iloc[j]+'.jpg'
            shutil.move(image_src, image_dst)
        else:
            image_src = path_src+'/'+group_temp['id'].iloc[j]+'.jpg'
            image_dst = path_val_dst+'/'+group_temp['id'].iloc[j]+'.jpg'
            shutil.move(image_src, image_dst)
    print(types[i]+'\t'+'finished')