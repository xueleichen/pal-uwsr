"""
   tools for progressive growing of sr model
"""
import tensorflow as tf
from keras.layers import Add, UpSampling2D, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Reshape, Multiply, Layer
from keras.models import Model
from keras import backend

n_residual_blocks = 8
gf = 64

def residual_block(layer_input, filters):
	"""Residual atentional block"""
	
	d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
	d = BatchNormalization(momentum=0.5)(d)
	d = Activation('relu')(d)
	d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
	
	h = GlobalAveragePooling2D()(d)
	h = Reshape((1,1,filters))(h)
	h = Conv2D(filters//8, kernel_size=1, strides=1, padding='same')(h)
	h = Activation('relu')(h)
	h = Conv2D(filters, kernel_size=1, strides=1, padding='same')(h)
	h = Activation('sigmoid')(h)
	h = Multiply()([d,h])

	hout = Add()([h, layer_input])
	return hout

def deconv2d(layer_input):
	"""Layers used during upsampling"""
	u = UpSampling2D(size=2)(layer_input)
	u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
	u = Activation('relu')(u)
	return u

def res_mult_2x(layer_input):
	l1 = Conv2D(64, kernel_size=4, strides=1, padding='same')(layer_input)
	l1 = Activation('relu')(l1)
	# Propogate through residual blocks
	r = residual_block(l1, gf)
	for _ in range(n_residual_blocks - 1):
		r = residual_block(r, gf)
	# Post-residual block
	l2 = Conv2D(64, kernel_size=4, strides=1, padding='same')(r)
	l2 = BatchNormalization(momentum=0.8)(l2)
	l2 = Add()([l2, l1])
	# Upsampling
	layer_2x = deconv2d(l2)
	return layer_2x

def update_fadein(model, step, n_steps):
	# calculate current alpha (linear from 0 to 1)
	alpha = step / float(n_steps - 1)
	# update the alpha for each model
	layer = model.layers[-1]
	layer.alpha.assign(alpha)
	
class WeightedSum(Layer):

	def __init__(self, alpha=0.0, **kwargs):
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = tf.Variable(alpha, trainable=False)

	def call(self, inputs):
		# only supports a weighted sum of two inputs
		assert (len(inputs) == 2)
		# ((1-a) * input1) + (a * input2)
		output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
		return output

def update_model(old_model):
	for layer in old_model.layers:
		layer.name = layer.name + str("_o")
	# 2x old output
	o_ups = UpSampling2D()(old_model.layers[-2].output)
	o_rgb = Conv2D(3, kernel_size=5, strides=1, padding='same', activation='tanh')(o_ups)

	# new block
	h = res_mult_2x(old_model.layers[-2].output)
	n_rgb = Conv2D(3, kernel_size=5, strides=1, padding='same', activation='tanh')(h)

	# # weighted sum of old and new output
	out = WeightedSum()([o_rgb, n_rgb])

	# define model
	new_model = Model(old_model.input, out)

	return new_model