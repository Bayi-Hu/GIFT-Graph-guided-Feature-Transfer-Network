#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Model.model_seq import ModelSeq
from Utils.utils import multihead_attention
from Utils.utils import res_layer

class ModelSeqTargetAttenGIFT(ModelSeq):
    def __init__(self, tensor_dict, train_config):
        super(ModelSeqTargetAttenGIFT, self).__init__(tensor_dict, train_config)

    def target_attention_layer(self):
        # Attention layer
        with tf.name_scope('target_attention_layer'):
            query = res_layer(self.item_embedding, dim=32, name="query")
            key = res_layer(self.opt_seq_embedding, dim=32, name="key")

            attended_embedding = multihead_attention(queries=tf.expand_dims(query, axis=1),
                                                     keys=key,
                                                     values=key,
                                                     key_masks=self.sequence_mask,
                                                     dropout_rate=self.train_config["dropout_rate"],
                                                     is_training=self.train_config["is_training"]
                                                     )

            attended_embedding = tf.squeeze(attended_embedding, axis=1)
            return attended_embedding


    def GIFT_layer(self):
        """
        Returns:

        """
        attended_embeddings = []
        with tf.name_scope("gift_layer"):
            for i in range(n_hop):
                query = res_layer(self.item_embedding, dim=32, name="query")
                key = res_layer(self.tensor_dict[""], dim=32, name="key")

                attended_embedding = multihead_attention(queries=tf.expand_dims(query, axis=1),
                                                         keys=key,
                                                         values=key,
                                                         key_masks=self.sequence_mask,
                                                         dropout_rate=self.train_config["dropout_rate"],
                                                         is_training=self.train_config["is_training"]
                                                         )
                attended_embedding = tf.squeeze(attended_embedding, axis=1)
                attended_embeddings.append(attended_embedding)

            return attended_embeddings


    def build(self):
        """
        override the build function
        """

        self.attended_embedding = self.target_attention_layer()
        self.GIFT_attended_embeddings = self.GIFT_layer()

        inp = tf.concat([self.item_embedding,
                         self.user_embedding,
                         self.seq_item_embedding_sum,
                         self.attended_embedding] +
                        self.GIFT_attended_embeddings, axis=1)
        self.build_fcn_net(inp)
        self.loss_op()