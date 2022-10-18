#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Model.DNN import ModelSeq
from Utils.utils import layer_norm
from Utils.utils import multihead_attention

class DIN(ModelSeq):
    def __init__(self, tensor_dict, train_config):
        super(DIN, self).__init__(tensor_dict, train_config)

        self.length = tensor_dict["length"]
        self.opt_seq_embedding = tensor_dict["opt_seq_embedding"]

        # User sequence
        self.sequence_mask = tf.sequence_mask(self.length, maxlen=100, name="sequence_mask")
        dim = self.opt_seq_embedding.get_shape()[-1]
        mask_2d = tf.tile(tf.expand_dims(self.sequence_mask, axis=2), multiples=[1, 1, dim])
        self.masked_opt_seq_embedding = self.opt_seq_embedding * tf.cast(mask_2d, tf.float32)  # convert bool to float
        self.seq_item_embedding_sum = tf.reduce_sum(self.masked_opt_seq_embedding, axis=1)

        # GIFT features
        self.gift_length = tensor_dict["gift_length"]
        self.gift_embedding = tensor_dict["gift_embedding"]
        self.gift_sequence_mask = tf.sequence_mask(self.gift_length, maxlen=50, name="sequence_mask")

    def target_attention_layer(self):
        # Attention layer
        with tf.name_scope('din_layer'):

            query = self.item_embedding
            key = self.opt_seq_embedding
            value = self.opt_seq_embedding

            attended_embedding = multihead_attention(queries=tf.expand_dims(query, axis=1),
                                                     keys=key,
                                                     values=value,
                                                     key_masks=self.sequence_mask,
                                                     dropout_rate=self.train_config["dropout_rate"],
                                                     is_training=self.train_config["is_training"]
                                                     )
        output = tf.squeeze(attended_embedding, axis=1)

        return layer_norm(output, "din_layer_norm")

    def build(self):
        """
        override the build function
        """
        self.gift_atten_embedding = self.GIFT_layer()
        self.din_atten_embedding = tf.squeeze(self.target_attention_layer(), axis=1)
        inp = tf.concat([self.item_embedding, self.user_embedding, self.seq_item_embedding_sum, self.attended_embedding], axis=1)
        self.build_fcn_net(inp)
        self.loss_op()

    def GIFT_layer(self):
        """
        Returns:

        """
        # attended_embeddings = []
        with tf.name_scope("gift_layer"):

            query = self.item_embedding
            key = self.gift_embedding

            if self.atten_strategy == "mlp":
                query = tf.expand_dims(query, axis=1)
                queries = tf.tile(query, [1, key.shape[1], 1])
                attention_all = tf.concat([queries, key, queries-key, queries*key], axis=-1)

                d_layer_1_all = tf.layers.dense(attention_all, 64, activation=tf.nn.relu, name="f1_att")
                d_layer_2_all = tf.layers.dense(d_layer_1_all, 32, activation=tf.nn.relu, name="f2_att")
                d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name="f3_att")
                d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, key.shape[1]])

                scores = d_layer_3_all
                key_masks = tf.expand_dims(self.gift_sequence_mask, 1)

                paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
                scores = tf.where(key_masks, scores, paddings)
                scores = tf.nn.softmax(scores, axis=2)
                output = tf.squeeze(tf.matmul(scores, key))

            elif self.atten_strategy == "dot":
                attended_embedding = multihead_attention(queries=tf.expand_dims(query, axis=1),
                                                         keys=key,
                                                         values=key,
                                                         num_heads=8,
                                                         key_masks=self.gift_sequence_mask,
                                                         dropout_rate=self.train_config["dropout_rate"],
                                                         is_training=self.train_config["is_training"],
                                                         reuse=False)
                output = tf.squeeze(attended_embedding, axis=1)

            else:
                raise ValueError("Please choose the right attention strategy. Current is "+ self.atten_strategy)

        return layer_norm(output, "layer_norm")
