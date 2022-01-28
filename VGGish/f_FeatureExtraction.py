
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
info = scipy.io.loadmat('R:\\CMPH-Windfarm Field Study\\Hallett\\Hallett_file_info.mat')


num_secs = 624
num_files = info['filelist'].shape[0]
# X = np.empty([num_files, 128])
result_conv1 = np.empty([1000, 32])

for i in range(0,1000): # range(1, num_files)
    try:
        # wav_file = 'R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\\Data set\\set8\\audio\\so%d.wav'%(i+1,)
        # input_batch = vggish_input.wavfile_to_examples(wav_file)
        # Produce a batch of log mel spectrogram examples.
        #wav_file = 'R:\\CMPH-Windfarm Field Study\\Hallett\\set1\\Recording-3.%d.pti'%(i+1,)
        name = info['filelist'][i]['name'][0][0]
        folder = info['filelist'][i]['folder'][0][0]
        file_name_i = folder + '\\' + name
        x, sr, t, d = ptiread(file_name_i)
        input_batch = vggish_input.waveform_to_examples(x[:,3], sr)
        #print(file_name_i)
    
    
       # np.testing.assert_equal(
        #    input_batch.shape,
         #   [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])
    
    
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
        
        result_conv1[i] = embedding_batch.mean(axis=-1).mean(axis=1).mean(axis=0)
        
        # X[i] = np.mean(embedding_batch, axis=0)
        # X = np.mean(embedding_batch, axis=0)
    
        # print('\nLooks Good To Me!\n')
        print(i)

        #np.savetxt("R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\\Deepfeature\\HL\\X%d.txt"%(i+1), X,
         #          delimiter=",")
        
    except:
        print("An exception occurred")

np.savetxt("R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\\Low_high_feature\\2.Hallett\\result_conv1.csv", result_conv1, delimiter=",")






