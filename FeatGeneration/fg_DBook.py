import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class FeatGenerator(object):

    def __init__(self, input_file):
        self.input_file = input_file
        self.feat_config = {
            "n_user": 25000,
            "d_user": 16,

            "n_item": 30000,
            "d_item": 16,

            "n_group": 4000,
            "d_group": 8,

            "n_location": 2000,
            "d_location": 8,

            "n_publisher": 2000,
            "d_publisher": 8,

            "n_author": 20000,
            "d_author": 16,

            "n_year": 100,
            "d_year": 8,

            "max_length": 50,
            "batch_size": 128,
            "epoch": 1
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
        gift_length = values[12]

        return label, user, item, groups, location, publisher, author, year, gift_item, gift_author, gift_publisher, gift_year, gift_length

    def parse_sequence(self, sequence, max_length, delimiter="", default_value="0"):
        """
        split the sequence and convert to dense tensor
        """
        split_sequence = tf.string_split(sequence, delimiter=delimiter)
        split_sequence = tf.sparse_to_dense(sparse_indices=split_sequence.indices,
                                            output_shape=[self.feat_config["batch_size"],
                                                          max_length],
                                            sparse_values=split_sequence.values, default_value=default_value)

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

        label, user, item, groups, location, publisher, author, year, gift_item, gift_author, gift_publisher, gift_year, gift_length = iterator.get_next()
        # Dbook dataset has no sequence feature

        gift_item = self.parse_sequence(gift_item, max_length=50, delimiter=",", default_value="nan")
        gift_author = self.parse_sequence(gift_author, max_length=50, delimiter=",", default_value="nan")
        gift_publisher = self.parse_sequence(gift_publisher, max_length=50, delimiter=",", default_value="nan")
        gift_year = self.parse_sequence(gift_year, max_length=50, delimiter=",", default_value="nan")

        features = {}
        features["label"] = tf.one_hot(tf.string_to_number(label, out_type=tf.int32), depth=2)
        features["user"] = user
        features["item"] = item
        features["groups"] = groups
        features["location"] = location
        features["author"] = author
        features["publisher"] = publisher
        features["year"] = year

        features["gift_item"] = gift_item
        features["gift_author"] = gift_author
        features["gift_publisher"] = gift_publisher
        features["gift_year"] = gift_year
        features["gift_length"] = tf.string_to_number(gift_length, out_type=tf.int32)

        return features


class TensorGenerator(object):

    def __init__(self):
        pass

    def embedding_layer(self, features, feat_config):

        with tf.name_scope('Embedding_layer'):

            user_lookup_table = tf.get_variable("user_embedding_var", [feat_config["n_user"], feat_config["d_user"]])
            item_lookup_table = tf.get_variable("item_embedding_var", [feat_config["n_item"], feat_config["d_item"]])
            group_lookup_table = tf.get_variable("group_embedding_var", [feat_config["n_group"], feat_config["d_group"]])
            location_lookup_table = tf.get_variable("location_embedding_var", [feat_config["n_location"], feat_config["d_location"]])
            publisher_lookup_table = tf.get_variable("publisher_embedding_var", [feat_config["n_publisher"], feat_config["d_publisher"]])
            author_lookup_table = tf.get_variable("author_embedding_var", [feat_config["n_author"], feat_config["d_author"]])
            year_lookup_table = tf.get_variable("year_embedding_var", [feat_config["n_year"], feat_config["d_year"]])

            # add to summary
            # tf.summary.histogram('user_lookup_table', user_lookup_table)

            # user feature
            user_embedding = tf.nn.embedding_lookup(user_lookup_table,
                                                    tf.string_to_hash_bucket_fast(features["user"],
                                                                                 feat_config["n_user"]))

            location_embedding = tf.nn.embedding_lookup(location_lookup_table,
                                                        tf.string_to_hash_bucket_fast(features["location"],
                                                                                      feat_config["n_location"]))

            # item feature
            item_embedding = tf.nn.embedding_lookup(item_lookup_table,
                                                    tf.string_to_hash_bucket_fast(features["item"],
                                                                                 feat_config["n_item"]))

            author_embedding = tf.nn.embedding_lookup(author_lookup_table,
                                                      tf.string_to_hash_bucket_fast(features["author"],
                                                                                    feat_config["n_author"]))

            publisher_embedding = tf.nn.embedding_lookup(publisher_lookup_table,
                                                         tf.string_to_hash_bucket_fast(features["publisher"],
                                                                                       feat_config["n_publisher"]))

            year_embedding = tf.nn.embedding_lookup(year_lookup_table,
                                                    tf.string_to_hash_bucket_fast(features["year"],
                                                                                  feat_config["n_year"]))


            # item sequence
            # seq_iid_embedding = tf.nn.embedding_lookup(iid_lookup_table,
            #                                            tf.string_to_hash_bucket_fast(features["seq_item_id"],
            #                                                                          feat_config["n_iid"]))
            #
            # seq_cat_embedding = tf.nn.embedding_lookup(cat_lookup_table,
            #                                            tf.string_to_hash_bucket_fast(features["seq_category"],
            #                                                                          feat_config["n_cid"]))

            # gift sequence
            gift_item_embedding = tf.nn.embedding_lookup(item_lookup_table,
                                                         tf.string_to_hash_bucket_fast(features["gift_item"],
                                                                                       feat_config["n_item"]))

            gift_author_embedding = tf.nn.embedding_lookup(author_lookup_table,
                                                           tf.string_to_hash_bucket_fast(features["gift_author"],
                                                                                         feat_config["n_author"]))

            gift_publisher_embedding = tf.nn.embedding_lookup(publisher_lookup_table,
                                                              tf.string_to_hash_bucket_fast(features["gift_publisher"],
                                                                                            feat_config["n_publisher"]))

            gift_year_embedding = tf.nn.embedding_lookup(year_lookup_table,
                                                         tf.string_to_hash_bucket_fast(features["gift_year"],
                                                                                       feat_config["n_year"]))


            # concatenate the tensors
            tensor_dict = {}
            tensor_dict["user_embedding"] = tf.concat([user_embedding, location_embedding], 1)
            tensor_dict["item_embedding"] = tf.concat([item_embedding, author_embedding, publisher_embedding, year_embedding], 1)

            # tensor_dict["opt_seq_embedding"] = tf.concat([seq_iid_embedding, seq_cat_embedding], 2)

            # gift feature
            tensor_dict["gift_embedding"] = tf.concat([gift_item_embedding, gift_author_embedding, gift_publisher_embedding, gift_year_embedding], 2)
            tensor_dict["gift_length"] = features["gift_length"]
            tensor_dict["label"] = features["label"]

        return tensor_dict

if __name__ == '__main__':

    train_file = "../FeatGeneration/DBook/ui_sample_gift_new.csv"

    train_fg = FeatGenerator(train_file)
    train_features = train_fg.feature_generation()

    tg = TensorGenerator()
    train_tensor_dict = tg.embedding_layer(train_features, train_fg.feat_config)
    print("pause")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(train_tensor_dict["user_embedding"])

