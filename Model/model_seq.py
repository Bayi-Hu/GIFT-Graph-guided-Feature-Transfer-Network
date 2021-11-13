#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model import Model


class ModelSeq(Model):
    def __init__(self, tensor_dict):
        super(ModelSeq, self).__init__(tensor_dict)

        # notice, it should be mask with the length mask ..
        self.sequence_mask = tf.sequence_mask(self.length, maxlen=100, name="sequence_mask")
        dim = self.opt_seq_embedding.get_shape()[-1]
        mask_2d = tf.tile(tf.expand_dims(self.sequence_mask, axis=2), multiples=[1, 1, dim])
        self.masked_opt_seq_embedding = self.opt_seq_embedding * mask_2d
        self.seq_item_embedding_sum = tf.reduce_sum(self.masked_opt_seq_embedding, axis=1)


    def build(self):
        """
        override the build function
        """
        inp = tf.concate([self.item_embedding, self.user_embedding, self.seq_item_embedding_sum], axis=1)
        self.build_fcn_net(inp)
        self.loss_op()
