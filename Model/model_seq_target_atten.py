#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model_seq import ModelSeq

class ModelSeqTargetAtten(ModelSeq):
    def __init__(self, tensor_dict):
        super(ModelSeqTargetAtten, self).__init__(tensor_dict)

    def target_attention_layer(self, query, key, value, length):
        # Attention layer
        with tf.name_scope('target_attention_layer'):
            # TODO

            return attended_embedding

    def build(self):
        """
        override the build function
        """
        self.attended_embedding = self.target_attention_layer(query=self.item_embedding,
                                                              key=self.opt_seq_embedding,
                                                              value=self.opt_seq_embedding,
                                                              length=self.length)

        inp = tf.concate([self.item_embedding, self.user_embedding, self.seq_item_embedding_mean, self.attended_embedding], axis=1)
        self.build_fcn_net(inp)
        self.loss_op()