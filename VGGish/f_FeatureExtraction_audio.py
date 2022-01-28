# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 21:19:47 2021

@author: nguy0936

I used this code to extract feature from audio files in benchmark files

"""


from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_slim

import pyenvnoise
from pyenvnoise.utils import ptiread
import scipy.io


print('\nTesting your install of VGGish\n')

# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'


# Load audio files from folder
#info = scipy.io.loadmat('R:\\CMPH-Windfarm Field Study\\Hallett\\Hallett_file_info.mat')


#num_secs = 624
#num_files = info['filelist'].shape[0]
#X = np.empty([num_files, 128])

#result_conv1 = np.empty([56364, 32])
#result_conv2 = np.empty([56364, 16])
#result_conv3 = np.empty([3000, 8])
result_conv4 = np.empty([3000, 4])
#result_embedding = np.empty([56364, 128])


for i in range(0,3000): # range(1, num_files)
    try: 
        wav_file = 'R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\\Data set\\set2\\audio\\s%d.wav'%(i+1,)
        input_batch = vggish_input.wavfile_to_examples(wav_file)
        
        # Define VGGish,load the checkpoint,and run the batch through the model to
        # produce embeddings.
        with tf.Graph().as_default(), tf.Session() as sess:
            vggish_slim.define_vggish_slim()
            vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
    
            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: input_batch})
            
        # print('VGGish embedding: ', embedding_batch[2])
        # print(embedding_batch.shape)
        
        #result_conv1[i] = embedding_batch.mean(axis=-1).mean(axis=1).mean(axis=0)
        #result_conv2[i] = embedding_batch.mean(axis=-1).mean(axis=1).mean(axis=0)
        #result_conv3[i] = embedding_batch.mean(axis=-1).mean(axis=1).mean(axis=0)
        result_conv4[i] = embedding_batch.mean(axis=-1).mean(axis=1).mean(axis=0)
        #result_embedding[i] = embedding_batch.mean(axis=0)
        print(i)
        
    except:
        print("An exception occurred")

np.savetxt("R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\\Data set\\set2\\result_conv4.csv", result_conv4, delimiter=",")


#conv1_slice1 = embedding_batch[0][:,:,20]
#imgplot = plt.imshow(embedding_batch[0][:,:,32])
#imgplot = plt.imshow(input_batch[0,:,:])
#np.save('result_conv4', embedding_batch)




