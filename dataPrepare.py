#!/usr/bin/python
# -*- coding: UTF-8 -*-


from data import *
data_gen_args = dict(rotation_range=0.1,
                    width_shift_range=0.01,
                    height_shift_range=0.01,
                    shear_range=0.01,
                    zoom_range=0.01,
                    horizontal_flip=False,
                    fill_mode='nearest')
num_batch = 1
myGenerator = trainGenerator(9,'data/qrqm/test','test_src','test_src',data_gen_args,target_size=(512,512),save_to_dir = "data/qrqm/test/test_src")
# myGenerator = trainGenerator(1,'data/membrane/train','image','label',data_gen_args,save_to_dir = "data/membrane/train/aug")
for i,batch in enumerate(myGenerator):
    if (i > num_batch):
        break