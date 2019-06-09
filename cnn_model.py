# -*- coding: utf-8 -*-
"""
Created on Tue May  7 08:48:54 2019

@author: ASUS
"""

import tensorflow as tf
import numpy as np


def cnn(test_X):
    model_path = "Model 2019-05-28_16.04.56/model.ckpt"
    
    
    
#    output = inference_graph.get_tensor_by_name('outputPred:0')
    
    
    with tf.Session() as sess:
       
        
        loader = tf.train.import_meta_graph(model_path+'.meta')
        loader.restore(sess,model_path)
        
        graph = tf.get_default_graph()
        
        pred = graph.get_tensor_by_name('outputPred:0')
        x = graph.get_tensor_by_name('x:0')
      
        
        output_class = sess.run(pred, feed_dict={x:test_X})
        
        
        
        sess.close()
    for x in output_class:
        print(np.amax(x))
        
    prob_class = np.copy(output_class)[:,0]
    
    output_class[output_class > 0.5] = 1
    output_class[output_class <= 0.5] = 0
    
    
    output_class = output_class.astype(int)   
    
    
    index = []
    output_class = output_class.tolist()
    
    for i in range(len(output_class)):
        if output_class[i] == [1,0]:
            index.append(i)
    
    prob_class = prob_class[index]
    
    index = np.array(index)
    
    pos = np.where(prob_class > 0.9999)
    return output_class,index[pos],prob_class[pos] 
        
    
    
    
   