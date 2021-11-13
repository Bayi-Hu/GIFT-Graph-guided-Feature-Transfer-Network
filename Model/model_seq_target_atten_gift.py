#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Model.model_seq_target_atten import ModelSeqTargetAtten

class ModelSeqTargetAttenGIFT(ModelSeqTargetAtten):
    def __init__(self, tensor_dict, train_config):
        super(ModelSeqTargetAttenGIFT, self).__init__(tensor_dict, train_config)


    def build(self):
        """
        override the build function
        """
        self.attended_embedding = self.target_attention_layer(query=self.item_embedding,
                                                              key=self.opt_seq_embedding,
                                                              value=self.opt_seq_embedding,
                                                              length=self.length)

        # gift part




        inp = tf.concat([self.item_embedding, self.user_embedding, self.seq_item_embedding_mean, self.attended_embedding], axis=1)
        self.build_fcn_net(inp)
        self.loss_op()


