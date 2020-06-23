#!/usr/bin/env python
"""
# > Script for training 8x generative SISR models on USR-248 data 
#    - Paper: https://arxiv.org/pdf/1909.09437.pdf
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
import os
import sys
import datetime
import numpy as np
# keras libs
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # less logs
# local libs
from utils.plot_utils import save_val_samples
from utils.data_utils import dataLoaderUSR, deprocess, dataLoaderUSR_ori
from utils.loss_utils import perceptual_distance, total_gen_loss
# network
from nets.gen_models import ASRDRM_gen
#############################################################################
## dataset and image information
channels = 3
lr_width, lr_height = 80, 60   # low res
hr_width, hr_height = 640, 480 # high res
# input and output data
lr_shape = (lr_height, lr_width, channels)
hr_shape = (hr_height, hr_width, channels)
data_loader = dataLoaderUSR_ori(DATA_PATH="/content/drive/My Drive/USR-248/", SCALE=2)

# training parameters
num_epochs = int(sys.argv[1])
batch_size = 2
sample_interval = 500 # per step
ckpt_interval = 5
steps_per_epoch = (data_loader.num_train//batch_size)
num_step = num_epochs*steps_per_epoch

model_name="attention_learning_2x"

###################################################################################
# initialize the model
model_loader = ASRDRM_gen(lr_shape, hr_shape, SCALE=8)
model = model_loader.create_model()
print (model.summary())
# checkpoint directory
checkpoint_dir = os.path.join("/content/drive/My Drive/USR/ablation/checkpoints/", model_name)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
## sample directory
samples_dir = os.path.join("/content/drive/My Drive/USR/ablation/images/", model_name)
if not os.path.exists(samples_dir): os.makedirs(samples_dir)

#####################################################################
# compile model

optimizer_ = Adam(0.0002, 0.5)

model.compile(optimizer=optimizer_, loss=total_gen_loss)

print ("\nTraining: {0}".format(model_name))
## training pipeline
step, epoch = 0, 0; start_time = datetime.datetime.now()
while (step <= num_step):
    for i, (imgs_lr, imgs_hr) in enumerate(data_loader.load_batch(batch_size)):
        # train the generator
        loss_i = model.train_on_batch(imgs_lr, imgs_hr)
        # increment step, and show the progress 
        step += 1; elapsed_time = datetime.datetime.now() - start_time
        if (step%10==0):
            print ("[Epoch %d: batch %d/%d] [loss_i: %f]" 
                               %(epoch, i+1, steps_per_epoch, loss_i))
        ## validate and save generated samples at regular intervals 
        if (step % sample_interval==0):
            imgs_lr, imgs_hr = data_loader.load_val_data(batch_size=2)
            fake_hr = model.predict(imgs_lr)
            gen_imgs = np.concatenate([deprocess(fake_hr), deprocess(imgs_hr)])
            save_val_samples(samples_dir, gen_imgs, step)

    epoch += 1

    if (epoch%ckpt_interval==0):
        ckpt_name = os.path.join(checkpoint_dir, ("model_%d" %epoch))
        with open(ckpt_name+"_.json", "w") as json_file:
            json_file.write(model.to_json())
        model.save_weights(ckpt_name+"_.h5")
        print("\nSaved trained model in {0}\n".format(checkpoint_dir))

