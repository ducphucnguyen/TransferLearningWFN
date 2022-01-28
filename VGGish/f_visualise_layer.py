# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:33:49 2021

@author: nguy0936
"""


import matplotlib.pyplot as plt
import numpy as np


## Input_batch layer
result_input = np.load('input_batch.npy')

for i in range(0, 10):
    plt.subplot(2, 5, i+1)
    plt.imshow( result_input[i,:,:], cmap='coolwarm', interpolation='nearest' )
    plt.axis('off')

plt.imshow( result_input[0,:,:], cmap='coolwarm', interpolation='nearest' )
plt.savefig('result_input.pdf')  

## Inspect checkpoint 
checkpoint_path = 'vggish_model.ckpt'  
  
from tensorflow.python.training import py_checkpoint_reader
reader = py_checkpoint_reader.NewCheckpointReader(checkpoint_path)

var_to_shape_map = reader.get_variable_to_shape_map()


var_to_dtype_map = reader.get_variable_to_dtype_map()

ckpt_vars = list(var_to_shape_map.keys())
filter1 = reader.get_tensor(ckpt_vars[1])

filter11 = filter1[:,:,0,0]
#imgplot = plt.imshow( filter1[:,:,0,1], cmap='gray', interpolation='nearest' )


## filter 1
for i in range(0, 64):
    plt.subplot(8, 8, i+1)
    plt.imshow( filter1[:,:,0,i], cmap='coolwarm', interpolation='bilinear' )
    plt.axis('off')

plt.tight_layout()
plt.savefig('64_filters.pdf')  
plt.show()



## Conv1
result_conv1 = np.load('result_conv1.npy')

for i in range(0, 64):
    plt.subplot(8, 8, i+1)
    plt.imshow( result_conv1[0,:,:,i], cmap='coolwarm', interpolation='nearest' )
    plt.axis('off')


plt.imshow( result_conv1[0,:,:,32], cmap='coolwarm', interpolation='nearest' )
plt.axis('off')
plt.savefig('conv1_vertical.pdf')  
plt.show()

plt.imshow( result_conv1[0,:,:,28], cmap='coolwarm', interpolation='nearest' )
plt.axis('off')
plt.savefig('conv1_horizontal.pdf')  



## Filter 2

filter2 = reader.get_tensor(ckpt_vars[3])


## filter 2
for i in range(0, 128):
    plt.subplot(16, 8, i+1)
    plt.imshow( filter2[:,:,31,i], cmap='coolwarm', interpolation='bilinear' )
    plt.axis('off')

plt.savefig('128_filter2.pdf')  



## Conv2
result_conv2 = np.load('result_conv2.npy')

for i in range(0, 128):
    plt.subplot(16, 8, i+1)
    plt.imshow( result_conv2[0,:,:,i], cmap='coolwarm', interpolation='nearest' )
    plt.axis('off')


plt.imshow( result_conv2[0,:,:,32], cmap='coolwarm', interpolation='nearest' )
plt.axis('off')
plt.savefig('conv2_vertical.pdf') 






## Conv3
result_conv3 = np.load('result_conv3.npy')

for i in range(0, 256):
    plt.subplot(32, 8, i+1)
    plt.imshow( result_conv3[0,:,:,i], cmap='coolwarm', interpolation='nearest' )
    plt.axis('off')


plt.imshow( result_conv3[0,:,:,248], cmap='coolwarm', interpolation='nearest' )
plt.axis('off')
plt.savefig('conv3_vertical.pdf') 



## Conv4
result_conv4 = np.load('result_conv4.npy')

for i in range(0, 512):
    plt.subplot(64, 8, i+1)
    plt.imshow( result_conv4[0,:,:,i], cmap='coolwarm', interpolation='nearest' )
    plt.axis('off')


plt.imshow( result_conv4[0,:,:,3], cmap='coolwarm', interpolation='nearest' )
plt.axis('off')
plt.savefig('conv4_vertical.pdf') 





