

from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Deconv2DLayer, flatten, reshape, batch_norm, Upscale2DLayer

from lasagne.nonlinearities import rectify as relu

from lasagne.nonlinearities import LeakyRectify as lrelu

from lasagne.nonlinearities import sigmoid, tanh

from lasagne.layers import get_output, get_all_params, get_output_shape, get_all_layers

from lasagne.objectives import binary_crossentropy as bce

from lasagne.updates import adam, sgd, rmsprop



import numpy as np

import theano

from theano import tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import time

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt

import scipy.io

import scipy.misc

from skimage.io import imsave



floatX=theano.config.floatX


# DEFINE PATHS 


inPath = 'path/dataset' # path to where your dataset is stored

outPath = 'path/results' # path to folder where the results will be saved



print('PACKAGES LOADED')

# define function to create montage of 100 images 

def create_montage(image):

	_,_,sx,sy=np.shape(image) 

	montage=np.ones(shape=(10*sx,10*sy,1)) 

	n,x,y = [0,0,0]



	for i in range(10): 

	    for j in range(10):  

	        im=image[n,:,:,:].transpose(1,2,0) 

	        n+=1 

	        montage[x:x+sx,y:y+sy,:]=im 

	        x+=sx 

	    x=0 

	    y+=sy 

	print 'montage:',np.shape(montage) 

	return montage



def get_args():

	print 'getting args...'



def save_args():

	print 'saving args...'


# DEFINE NETWORK

def build_net(nz=200):



	gen = InputLayer(shape=(None,nz))

	gen = DenseLayer(incoming=gen, num_units=1024*4*4)

	gen = reshape(incoming=gen, shape=(-1,1024,4,4))

	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=512, filter_size=4, stride=2, nonlinearity=relu, crop=1))

	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=256, filter_size=4, stride=2, nonlinearity=relu, crop=1))

	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=128, filter_size=4, stride=2, nonlinearity=relu, crop=1))

	gen = Deconv2DLayer(incoming=gen, num_filters=1, filter_size=4, stride=2, nonlinearity=sigmoid, crop=1) ### or tanh?


	dis = InputLayer(shape=(None,1,64,64))

	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=128, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))

	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=256, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))

	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=512, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)) 

	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=1024, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)) 

	dis = reshape(incoming=dis, shape=(-1,1024*4*4))

	dis = DenseLayer(incoming=dis, num_units=1, nonlinearity=None)

	return gen, dis


G,D=build_net(nz=200)

# print out the shape of the network at every layer

for l in get_all_layers(G):

        print get_output_shape(l)

for m in get_all_layers(D):

	print get_output_shape(m)
        


def prep_train(alpha=0.002, beta1=0.5, beta2=0.9, nz=200):

	G,D=build_net(nz=nz)

	x = T.tensor4('x')

	z = T.matrix('z')

	# get network output for D and G

	G_z=get_output(G,z)

	D_G_z=get_output(D,G_z) # fake

	D_x=get_output(D,x) # real

	# create new variable e to sample X along straight lines

	e = T.TensorType(dtype=floatX,broadcastable=(False, True, True, True))()

	mixed_X = (e * G_z) + (1-e) * x

	output_D_mixed=get_output(D,mixed_X)

	#compute gradients + penalty

	grad_mixed=T.grad(T.sum(output_D_mixed),mixed_X)
	
	norm_grad_mixed=T.sqrt(T.sum(T.square(grad_mixed),axis=[1,2,3]))
	
	grad_penalty = T.mean(T.square(norm_grad_mixed -1))

	# get parameters

	params_d=get_all_params(D, trainable=True) 

	params_g=get_all_params(G, trainable=True) 

	# compute losses for the discriminator J_D and the generator J_G

	J_D = D_G_z.mean() - D_x.mean() + 10 * grad_penalty

	J_G = - D_G_z.mean()

	# update parameters for both

	update_D = adam(J_D,params_d, learning_rate=alpha, beta1=beta1, beta2=beta2) 

	update_G = adam(J_G,params_g, learning_rate=alpha, beta1=beta1, beta2=beta2) 

	# define training functions

	train_G=theano.function(inputs=[z], outputs=J_G, updates=update_G) 

	train_D=theano.function(inputs=[x,z,e], outputs=J_D, updates=update_D)



	return train_G, train_D, G, D


def train(trainData, nz=200, alpha=0.00005, beta1=0.5 , beta2=0.9, batchSize=128, epoch=400): 


	train_G, train_D, G, D = prep_train(nz=nz, alpha=alpha, beta1=beta1, beta2=beta2) 

	sn,sc,sx,sy=np.shape(trainData) 

	print sn,sc,sx,sy

	batches=int(np.floor(float(sn)/batchSize)) 

	g_cost=np.zeros(epoch*batches)

	d_cost=np.zeros(epoch*batches)

	print 'batches=',batches

	timer=time.time() 

	print 'epoch \t\t batch \t\t cost G \t\t cost D \t\t time (s)'

	for e in range(epoch):

		for b in range(batches):

			Z = np.random.normal(loc=0.0, scale=1.0, size=(sn,nz)).astype(floatX)

			random_epsilon = np.random.uniform(size=(batchSize, 1,1,1)).astype('float32')

			cost_D=train_D(trainData[b*batchSize:(b+1)*batchSize],Z[b*batchSize:(b+1)*batchSize],random_epsilon)  

			cost_G=train_G(Z[b*batchSize:(b+1)*batchSize])

			# save results every 25 iterations within batch

			if b%25==0:

				A=test(G).eval() # evaluate network

				montage_gen = create_montage(A) # create montage of fake images	

				# save montage

				imsave(outPath + 'gen_{}.png'.format(e), montage_gen[:,:,0])


			print e,'\t\t',b,'\t\t',cost_G,'\t\t', cost_D,'\t\t', time.time()-timer

			timer=time.time()

			# append losses

			g_cost[e*batches + b] = cost_G

			d_cost[e*batches + b] = cost_D

			# create plot of loss functions that gets updated at every iteration

			plt.plot(range(e*batches+b),g_cost[:e*batches + b],color='magenta') # update generator loss 

			plt.plot(range(e*batches+b),d_cost[:e*batches + b],color='green') # udpate discriminator loss

			plt.legend()

			plt.xlabel('Iterations')

			plt.ylabel('Cost Function')

			plt.savefig(outPath + 'cost_regular_{}.png'.format(e))


	return G, D


# define function to test the generator 

def test(G):

	Z=np.random.normal(loc=0.0, scale=1.0, size=(100,200)).astype(floatX)

	G_Z=get_output(G,Z,deterministic=True)

	return G_Z


# define function that loads the dataset

def loadData():

    data=np.load(inPath,mmap_mode='r').astype(floatX)

    # check whether the min and max are between 0 and 1

    print 'Data min and max:', data.min(), data.max()

    #np.random.shuffle(data)

    return data


print('DATA LOADING')

x_train=loadData()

print(np.shape(x_train))

print('DATA LOADED')

print('TRAINING MODEL')

G,D=train(x_train)
