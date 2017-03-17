# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from YOLO_small_tf import *

yolo_tf = YOLO_TF()
with yolo_tf.sess.as_default():
    for v in tf.trainable_variables():
        name = v.name
        ary = v.eval()
        print name, np.shape(ary)
        np.save("npy/"+name, ary)
