import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class FeatGenerator(object):

    def __init__(self, input_file):
        self.input_file = input_file
        self.feat_config = {
            "n_user": 1000,
            "d_user": 32,

            "n_item": 1000,
            "d_item": 32,

            "n_group": 1000,
            "d_group": 32,

            "n_location": 1000,
            "d_location": 32,

            "n_publisher": 1000,
            "d_publisher": 32,

            "n_author": 1000,
            "d_author": 32,

            "n_year": 1000,
            "d_year": 32,

            "max_length": 50,
            "batch_size": 128,
            "epoch": 3
        }

    def parse_split(self, line):
        parse_res = tf.string_split([line], delimiter="\t")
        values = parse_res.values
        user = values[0]
        item = values[1]
        label = values[2]

        groups = values[3]
        location = values[4]

        publisher = values[5]
        author = values[6]
        year = values[7]

        gift_item = values[8]
        gift_author = values[9]
        gift_publisher = values[10]
        gift_year = values[11]

        length = values[6]

        return label, user_id, item_id, category, iid_sequence, cat_sequence, length

    def parse_sequence(self, sequence):
        """
        split the sequence and convert to dense tensor
        """
        split_sequence = tf.string_split(sequence, delimiter="")
        split_sequence = tf.sparse_to_dense(sparse_indices=split_sequence.indices,
                                            output_shape=[self.feat_config["batch_size"],
                                                          self.feat_config["max_length"]],
                                            sparse_values=split_sequence.values, default_value="0")

        return split_sequence

    def feature_generation(self):
        """
        Args:
            input_file: a .txt file that includes the training or testing sample
        Returns:
            feature tensor used for training or testing
        """
        dataset = tf.data.TextLineDataset(self.input_file)
        dataset = dataset.map(self.parse_split, num_parallel_calls=2)
        dataset = dataset.shuffle(3).repeat(self.feat_config["epoch"]).batch(self.feat_config["batch_size"])
        iterator = dataset.make_one_shot_iterator()

        label, user_id, item_id, category, seq_item_id, seq_category, length = iterator.get_next()

        seq_item_id = self.parse_sequence(seq_item_id)
        seq_category = self.parse_sequence(seq_category)

        features = {}
        features["label"] = tf.one_hot(tf.string_to_number(label, out_type=tf.int32), depth=2)
        features["user_id"] = user_id
        features["item_id"] = item_id
        features["category"] = category
        features["seq_item_id"] = seq_item_id
        features["seq_category"] = seq_category
        features["length"] = tf.string_to_number(length, out_type=tf.int32)

        return features

class TensorGenerator(object):

    def __init__(self):
        pass

    def embedding_layer(self, features, feat_config):

        with tf.name_scope('Embedding_layer'):

            uid_lookup_table = tf.get_variable("uid_embedding_var", [feat_config["n_uid"], feat_config["d_uid"]])
            iid_lookup_table = tf.get_variable("iid_embedding_var", [feat_config["n_iid"], feat_config["d_iid"]])
            cat_lookup_table = tf.get_variable("cat_embedding_var", [feat_config["n_cid"], feat_config["d_cid"]])

            # add to summary
            # tf.summary.histogram('uid_lookup_table', uid_lookup_table)
            # tf.summary.histogram('iid_lookup_table', iid_lookup_table)
            # tf.summary.histogram('cat_lookup_table', cat_lookup_table)

            uid_embedding = tf.nn.embedding_lookup(uid_lookup_table,
                                                   tf.string_to_hash_bucket_fast(features["user_id"],
                                                                                 feat_config["n_uid"]))

            iid_embedding = tf.nn.embedding_lookup(iid_lookup_table,
                                                   tf.string_to_hash_bucket_fast(features["item_id"],
                                                                                 feat_config["n_iid"]))

            cat_embedding = tf.nn.embedding_lookup(cat_lookup_table,
                                                   tf.string_to_hash_bucket_fast(features["category"],
                                                                                 feat_config["n_cid"]))

            # item sequence
            seq_iid_embedding = tf.nn.embedding_lookup(iid_lookup_table,
                                                       tf.string_to_hash_bucket_fast(features["seq_item_id"],
                                                                                     feat_config["n_iid"]))

            seq_cat_embedding = tf.nn.embedding_lookup(cat_lookup_table,
                                                       tf.string_to_hash_bucket_fast(features["seq_category"],
                                                                                     feat_config["n_cid"]))

            # concatenate the tensors
            tensor_dict = {}
            tensor_dict["user_embedding"] = uid_embedding
            tensor_dict["item_embedding"] = tf.concat([iid_embedding, cat_embedding], 1)
            tensor_dict["opt_seq_embedding"] = tf.concat([seq_iid_embedding, seq_cat_embedding], 2)
            tensor_dict["length"] = features["length"]
            tensor_dict["label"] = features["label"]

        return tensor_dict