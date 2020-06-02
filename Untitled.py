#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
from main import *
from model import *

model_filepath = "./pretrained/master.pb"

videofile = ''


tf.reset_default_graph()


# In[2]:


def main():
    
    cam = cv2.VideoCapture(0)
    model = GAN(model_filepath = model_filepath)
    cap = cv2.VideoCapture(videofile)
    
    while (cap.isOpened()):
        #ret_val, image = cam.read()
        ret, frame = cap.read()
            
        if ret:
            
            image = model.test(img = frame)
            
            cv2.imshow('my webcam', image/255.0)
            cv2.imwrite("o.bmp", image)
        
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    #cv2.destroyAllWindows()


# In[3]:


class GAN(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath = self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)

        with self.graph.as_default():
        	# Define input tensor
        	self.input = tf.placeholder(np.float32, shape = [1, None, None, 3], name='Placeholder')
        	tf.import_graph_def(graph_def, {'Placeholder': self.input})

        self.graph.finalize()

        print('Model loading complete!')

        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)
        
        """
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            # print("Value - " )
            # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        """

        # In this version, tf.InteractiveSession and tf.Session could be used interchangeably. 
        # self.sess = tf.InteractiveSession(graph = self.graph)
        self.sess = tf.Session(graph = self.graph)

    def test(self, img):
        
        img = (img-127.5)/127.5
        h,w = img.shape[:2]
        #input = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_CUBIC)
        #input= input.reshape(1, 720, 1280, 3)
        input= img.reshape(1, h, w, 3)

        # Know your output node name
        #data_np_expanded = np.expand_dims(data, axis=0)
        output_tensor = self.graph.get_tensor_by_name("import/output_image:0")
        output = self.sess.run(output_tensor, feed_dict = {self.input: input})
        #Y_ = output.reshape(720*4,1280*4,3)
        Y_ = output.reshape(h*4,w*4,3)
        Y_ = (Y_ + 1)*127.5
        
        print("output = ", Y_.shape)

        return Y_


# In[ ]:


if __name__ == '__main__':
    main()


# In[ ]:




