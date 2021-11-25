#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Model.model import Model
from Utils.utils import res_layer
from Utils.utils import multihead_attention

class ModelGIFT(Model):
    def __init__(self, tensor_dict, train_config):
        super(ModelGIFT, self).__init__(tensor_dict, train_config)

        self.gift_ia_length = tensor_dict["gift_ia_length"]
        self.gift_ia_embedding = tensor_dict["gift_ia_embedding"]
        self.gift_id_length = tensor_dict["gift_id_length"]
        self.gift_id_embedding = tensor_dict["gift_id_embedding"]

        # notice, it should be mask with the length mask
        # gift_ia
        self.gift_ia_sequence_mask = tf.sequence_mask(self.gift_ia_length, maxlen=30, name="sequence_mask")
        ia_dim = self.gift_ia_embedding.get_shape()[-1]
        ia_mask_2d = tf.tile(tf.expand_dims(self.gift_ia_sequence_mask, axis=2), multiples=[1, 1, ia_dim])
        self.masked_gift_ia_embedding = self.gift_ia_embedding * tf.cast(ia_mask_2d, tf.float32)
        self.gift_ia_embedding_sum = tf.reduce_sum(self.masked_gift_ia_embedding, axis=1)

        # gift_id
        self.gift_id_sequence_mask = tf.sequence_mask(self.gift_id_length, maxlen=30, name="sequence_mask")
        id_dim = self.gift_id_embedding.get_shape()[-1]
        id_mask_2d = tf.tile(tf.expand_dims(self.gift_id_sequence_mask, axis=2), multiples=[1, 1, id_dim])
        self.masked_gift_id_embedding = self.gift_id_embedding * tf.cast(id_mask_2d, tf.float32)
        self.gift_id_embedding_sum = tf.reduce_sum(self.masked_gift_id_embedding, axis=1)

    def GIFT_layer(self):
        """
        Returns:

        """
        # attended_embeddings = []
        with tf.name_scope("gift_layer"):
            # for i in range(n_hop):
            query = res_layer(self.item_embedding, dim=32, name="query")
            key = res_layer(self.gift_ia_embedding, dim=32, name="key_ia")

            ia_attended_embedding = multihead_attention(queries=tf.expand_dims(query, axis=1),
                                                         keys=key,
                                                         values=key,
                                                         key_masks=self.gift_ia_sequence_mask,
                                                         dropout_rate=self.train_config["dropout_rate"],
                                                         is_training=self.train_config["is_training"],
                                                         scope="gift_ia_attention",
                                                         reuse=False
                                                         )
            ia_attended_embedding = tf.squeeze(ia_attended_embedding, axis=1)

            key = res_layer(self.gift_id_embedding, dim=32, name="key_id")
            id_attended_embedding = multihead_attention(queries=tf.expand_dims(query, axis=1),
                                                        keys=key,
                                                        values=key,
                                                        key_masks=self.gift_id_sequence_mask,
                                                        dropout_rate=self.train_config["dropout_rate"],
                                                        is_training=self.train_config["is_training"],
                                                        scope="gift_id_attention",
                                                        reuse=False
                                                        )
            id_attended_embedding = tf.squeeze(id_attended_embedding, axis=1)

        return ia_attended_embedding, id_attended_embedding
            # attended_embeddings.append(attended_embedding)
        # return attended_embeddings

    def build(self):
        """
        override the build function
        """
        # self.gift_attended_embeddings = self.GIFT_layer()

        self.gift_ia_attended_embedding, self.gift_id_attended_embedding = self.GIFT_layer()
        inp = tf.concat([self.item_embedding,
                         self.user_embedding,
                         self.gift_ia_embedding_sum,
                         self.gift_id_embedding_sum,
                         self.gift_ia_attended_embedding,
                         self.gift_id_attended_embedding], axis=1)

        # inp = tf.concat([self.item_embedding,
        #                  self.user_embedding,
        #                  self.gift_embedding_sum], axis=1)

        # inp = tf.concat([self.item_embedding,
        #                  self.user_embedding], axis=1)

        self.build_fcn_net(inp)
        self.loss_op()
