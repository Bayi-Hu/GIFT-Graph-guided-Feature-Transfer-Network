#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model_seq import ModelSeq
from Utils.utils import multihead_attention
from Utils.utils import res_layer

class ModelSeqTargetAtten(ModelSeq):
    def __init__(self, tensor_dict):
        super(ModelSeqTargetAtten, self).__init__(tensor_dict)

    def target_attention_layer(self):
        # Attention layer
        with tf.name_scope('target_attention_layer'):
            query = res_layer(self.item_embedding, dim=32, name="query")
            key = res_layer(self.opt_seq_embedding, dim=32, name="key")
            value = res_layer(self.opt_seq_embedding, dim=32, name="value")

            attended_embedding = multihead_attention(queries=query,
                                                     keys=key,
                                                     values=value,
                                                     key_masks=self.sequence_mask,
                                                     dropout_rate=self.train_config["dropout_rate"],
                                                     is_training=self.train_config["is_training"]
                                                     )

            return attended_embedding

    def build(self):
        """
        override the build function
        """
        self.attended_embedding = self.target_attention_layer()
        inp = tf.concate([self.item_embedding, self.user_embedding, self.seq_item_embedding_sum, self.attended_embedding], axis=1)
        self.build_fcn_net(inp)
        self.loss_op()