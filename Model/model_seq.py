#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model import Model

class ModelSeq(Model):
    def __init__(self, tensor_dict):
        super(ModelSeq, self).__init__(tensor_dict)

        # notice, it should be mask with the length mask ..
        self.seq_item_embedding_mean = tf.reduce_mean(self.opt_seq_embedding, axis=1)

    def build(self):
        """
        override the build function
        """
        inp = tf.concate([self.item_embedding, self.user_embedding, self.seq_item_embedding_mean], axis=1)
        self.build_fcn_net(inp)
        self.loss_op()
