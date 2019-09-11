from model import *
from data import *
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
# myGene = trainGenerator(200,'data/qrqm/train','image','label',data_gen_args,save_to_dir = 'data/qrqm/train/aug')
myGene = trainGenerator(5,'data/qrqm/train','image','label',data_gen_args,target_size=(512,512),save_to_dir = None)
# myGene = trainGenerator(200,'data/qrqm/test','test_1','test_1',data_gen_args,save_to_dir = 'data/qrqm/test/aug')
model = unet(input_size=(512,512,1))
model_checkpoint = ModelCheckpoint('unet_membrane_512.hdf5', monitor='loss',verbose=1, save_best_only=True)
model = load_model('unet_membrane_512.hdf5')
model.fit_generator(myGene,steps_per_epoch=2000,epochs=20,callbacks=[model_checkpoint])