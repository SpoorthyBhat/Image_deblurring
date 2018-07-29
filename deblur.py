from model import Model
import preprocess
import cv2

"""
Parameters to be set:
"""

data_path = 'data'
training_size = 2000
dev_size = 20
epochs = 200
batch_size = 128
model_name = 'model.json'
weights_name = "model.h5"

#Create training ad dev set
X_train, Y_train, X_dev, Y_dev = preprocess.create_training_dev_set('unlabeled_images', 2000,10)

deblur = Model()
deblur.build_model()
deblur.train_model(epochs,batch_size,X_train,X_dev,Y_train,Y_dev)
deblur.save_model(model_name, weights_name)

deblur.predict_deblur('blur_lenna.png')

