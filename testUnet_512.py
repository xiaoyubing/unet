from model import *
from data import *
from keras.callbacks import ModelCheckpoint

img_dir = "data/qrqm/test/test_src_512"

testGene = testGenerator(img_dir,target_size=(512,512))
model = unet(input_size=(512,512,1))
model.load_weights("unet_membrane_512.hdf5")
results = model.predict_generator(testGene,9,verbose=1)
saveResult(img_dir,results)