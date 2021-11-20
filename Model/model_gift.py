#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Model.model import Model
from Utils.utils import res_layer
from Utils.utils import multihead_attention

class ModelGIFT(Model):
    def __init__(self, tensor_dict, train_config):
        super(ModelGIFT, self).__init__(tensor_dict, train_config)

        self.gift_length = tensor_dict["gift_length"]
        self.gift_embedding = tensor_dict["gift_embedding"]

        # notice, it should be mask with the length mask ..
        self.gift_sequence_mask = tf.sequence_mask(self.gift_length, maxlen=100, name="sequence_mask")
        dim = self.gift_embedding.get_shape()[-1]
        mask_2d = tf.tile(tf.expand_dims(self.gift_sequence_mask, axis=2), multiples=[1, 1, dim])
        self.masked_gift_embedding = self.gift_sequence_mask * tf.cast(mask_2d, tf.float32) # convert bool to float
        self.gift_embedding_sum = tf.reduce_sum(self.masked_gift_embedding, axis=1)

    def GIFT_layer(self):
        """
        Returns:

        """
        # attended_embeddings = []
        with tf.name_scope("gift_layer"):
            # for i in range(n_hop):
            query = res_layer(self.item_embedding, dim=32, name="query")
            key = res_layer(self.gift_embedding, dim=32, name="key")

            attended_embedding = multihead_attention(queries=tf.expand_dims(query, axis=1),
                                                     keys=key,
                                                     values=key,
                                                     key_masks=self.gift_sequence_mask,
                                                     dropout_rate=self.train_config["dropout_rate"],
                                                     is_training=self.train_config["is_training"]
                                                     )
            attended_embedding = tf.squeeze(attended_embedding, axis=1)

        return attended_embedding
            # attended_embeddings.append(attended_embedding)
        # return attended_embeddings

    def build(self):
        """
        override the build function
        """
        # self.gift_attended_embeddings = self.GIFT_layer()

        self.gift_attended_embedding = self.GIFT_layer()
        inp = tf.concat([self.item_embedding,
                         self.user_embedding,
                         self.gift_embedding_sum,
                         self.gift_attended_embedding], axis=1)

        self.build_fcn_net(inp)
        self.loss_op()
