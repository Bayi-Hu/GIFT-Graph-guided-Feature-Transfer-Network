import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class FeatGenerator(object):

    def __init__(self, input_file):
        self.input_file = input_file
        self.feat_config = {
            "n_user": 10000,
            "d_user": 16,

            "n_item": 5000,
            "d_item": 16,

            "n_genre": 30,
            "d_genre": 8,

            "n_age": 10,
            "d_age": 4,

            "n_occupation": 30,
            "d_occupation": 8,

            "n_zip": 4000,
            "d_zip": 16,

            "n_rating": 10,
            "d_rating": 4,

            "n_director": 3000,
            "d_director": 16,

            "n_actor": 9000,
            "d_actor": 16,

            "max_gift_ia_length": 50,
            "max_gift_id_length": 50,

            "max_length": 50,
            "batch_size": 128,
            "epoch": 4
        }

    def parse_split(self, line):
        parse_res = tf.string_split([line], delimiter="\t")
        values = parse_res.values

        user = values[0]
        item = values[1]
        # timestamp = values[2]
        label = values[3]

        gender = values[4]
        age = values[5]

        occupation = values[6]
        zip = values[7]
        rating = values[8]

        genre = values[9]
        director = values[10]
        actor = values[11]

        item_seq = values[12]
        rating_seq = values[13]
        genre_seq = values[14]
        length = values[15]

        # gift feature
        gift_ia_item = values[16]
        gift_ia_rating = values[17]
        gift_ia_genre = values[18]
        gift_ia_director = values[19]
        gift_ia_actor = values[20]
        gift_ia_length = values[21]

        gift_id_item = values[22]
        gift_id_rating = values[23]
        gift_id_genre = values[24]
        gift_id_director = values[25]
        gift_id_actor = values[26]
        gift_id_length = values[27]

        return label, user, item, gender, age, occupation, zip, rating, genre, director, actor, item_seq, rating_seq, genre_seq, length, \
               gift_ia_item, gift_ia_rating, gift_ia_genre, gift_ia_director, gift_ia_actor, gift_ia_length, \
               gift_id_item, gift_id_rating, gift_id_genre, gift_id_director, gift_id_actor, gift_id_length

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

        label, user, item, gender, age, occupation, zip, rating, genre, director, actor, seq_item, seq_rating, seq_genre, length, \
        gift_ia_item, gift_ia_rating, gift_ia_genre, gift_ia_director, gift_ia_actor, gift_ia_length, \
        gift_id_item, gift_id_rating, gift_id_genre, gift_id_director, gift_id_actor, gift_id_length = iterator.get_next()

        #
        print("pause")

        genre = tf.string_split(genre, delimiter="")
        genre = tf.SparseTensor(indices=genre.indices, values=tf.string_to_number(genre.values, out_type=tf.int32),
                                dense_shape=genre.dense_shape) # convert string to int32

        actor = tf.string_split(actor, delimiter="")
        actor = tf.SparseTensor(indices=actor.indices, values=tf.string_to_number(actor.values, out_type=tf.int32),
                                dense_shape=actor.dense_shape)  # convert string to int32

        director = tf.string_split(director, delimiter="")
        director = tf.SparseTensor(indices=director.indices, values=tf.string_to_number(director.values, out_type=tf.int32),
                                   dense_shape=director.dense_shape)  #

        # sequence
        SEQ_MAX_LENGTH = 50
        seq_item = self.parse_sequence(seq_item, max_length=SEQ_MAX_LENGTH, delimiter=",", default_value="nan")
        seq_rating = self.parse_sequence(seq_rating, max_length=SEQ_MAX_LENGTH, delimiter=",", default_value="nan")
        seq_genre = self.parse_sequence(seq_genre, max_length=SEQ_MAX_LENGTH, delimiter=",", default_value="nan")

        print("pause")
        seq_genre_flat = tf.reshape(seq_genre, shape=[128*50])
        seq_genre_split = tf.string_split(seq_genre_flat, delimiter="")

        def nan_convert(x):
            if tf.equal(x, "nan"):
                return "-1"
            else:
                return x

        seq_genre_split_values = tf.map_fn(nan_convert, seq_genre_split.values)
        seq_genre_new = tf.SparseTensor(indices=seq_genre_split.indices, values=tf.string_to_number(seq_genre_split_values, out_type=tf.int32),
                                        dense_shape=seq_genre_split.dense_shape)  # convert string to int32

        tf.sparse_reshape(seq_genre_new, shape=[128,50])

        # gift sequence
        # ia
        GIFT_IA_MAX_LENGTH = 50
        gift_ia_item = self.parse_sequence(gift_ia_item, max_length=GIFT_IA_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_ia_rating = self.parse_sequence(gift_ia_rating, max_length=GIFT_IA_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_ia_genre = self.parse_sequence(gift_ia_genre, max_length=GIFT_IA_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_ia_director = self.parse_sequence(gift_ia_director, max_length=GIFT_IA_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_ia_actor = self.parse_sequence(gift_ia_actor, max_length=GIFT_IA_MAX_LENGTH, delimiter=",", default_value="nan")

        # id
        GIFT_ID_MAX_LENGTH = 30
        gift_id_item = self.parse_sequence(gift_id_item, max_length=GIFT_ID_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_id_rating = self.parse_sequence(gift_id_rating, max_length=GIFT_ID_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_id_genre = self.parse_sequence(gift_id_genre, max_length=GIFT_ID_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_id_director = self.parse_sequence(gift_id_director, max_length=GIFT_ID_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_id_actor = self.parse_sequence(gift_id_actor, max_length=GIFT_ID_MAX_LENGTH, delimiter=",", default_value="nan")

        features = {}

        features["label"] = tf.one_hot(tf.string_to_number(label, out_type=tf.int32), depth=2)
        features["user"] = user
        features["item"] = item
        features["gender"] = gender
        features["age"] = age
        features["occupation"] = occupation
        features["zip"] = zip
        features["genre"] = genre
        features["rating"] = rating
        features["director"] = director
        features["actor"] = actor

        # sequence
        features["seq_item"] = seq_item
        features["seq_rating"] = seq_rating
        features["seq_genre"] = seq_genre
        features["length"] = length

        # gift_sequence
        features["gift_ia_item"] = gift_ia_item
        features["gift_ia_rating"] = gift_ia_rating
        features["gift_ia_genre"] = gift_ia_genre
        features["gift_ia_actor"] = gift_ia_actor
        features["gift_ia_director"] = gift_ia_director
        features["gift_ia_length"] = tf.string_to_number(gift_ia_length, out_type=tf.int32)

        features["gift_id_item"] = gift_id_item
        features["gift_id_rating"] = gift_id_rating
        features["gift_id_genre"] = gift_id_genre
        features["gift_id_actor"] = gift_id_actor
        features["gift_id_director"] = gift_id_director
        features["gift_id_length"] = tf.string_to_number(gift_id_length, out_type=tf.int32)

        return features

class TensorGenerator(object):

    def __init__(self):
        pass

    def embedding_layer(self, features, feat_config):

        with tf.name_scope('Embedding_layer'):

            user_lookup_table = tf.get_variable("user_embedding_var", [feat_config["n_user"], feat_config["d_user"]])
            item_lookup_table = tf.get_variable("item_embedding_var", [feat_config["n_item"], feat_config["d_item"]])

            rating_lookup_table = tf.get_variable("rating_embedding_var", [feat_config["n_rating"], feat_config["d_rating"]])
            age_lookup_table = tf.get_variable("age_embedding_var", [feat_config["n_age"], feat_config["d_age"]])
            occupation_lookup_table = tf.get_variable("occupation_embedding_var", [feat_config["n_occupation"], feat_config["d_occupation"]])
            zip_lookup_table = tf.get_variable("zip_embedding_var", [feat_config["n_zip"], feat_config["d_zip"]])
            genre_look_table = tf.get_variable("genre_embedding_var", [feat_config["n_genre"], feat_config["d_zip"]])
            actor_look_table = tf.get_variable("actor_embedding_var", [feat_config["n_actor"], feat_config["d_actor"]])
            director_look_table = tf.get_variable("director_embedding_var", [feat_config["n_director"], feat_config["d_director"]])

            # add to summary
            # tf.summary.histogram('user_lookup_table', user_lookup_table)

            # user feature
            user_embedding = tf.nn.embedding_lookup(user_lookup_table,
                                                    tf.string_to_hash_bucket_fast(features["user"],
                                                                                  feat_config["n_user"]))
            age_embedding = tf.nn.embedding_lookup(age_lookup_table,
                                                    tf.string_to_hash_bucket_fast(features["age"],
                                                                                  feat_config["n_age"]))
            zip_embedding = tf.nn.embedding_lookup(zip_lookup_table,
                                                   tf.string_to_hash_bucket_fast(features["zip"],
                                                                                 feat_config["n_age"]))
            # item feature
            item_embedding = tf.nn.embedding_lookup(item_lookup_table,
                                                    tf.string_to_hash_bucket_fast(features["item"],
                                                                                 feat_config["n_item"]))
            rating_embedding = tf.nn.embedding_lookup(rating_lookup_table,
                                                      tf.string_to_hash_bucket_fast(features["rating"],
                                                                                    feat_config["n_rating"]))

            genre_embedding = tf.nn.embedding_lookup_sparse(genre_look_table, sp_ids=features["genre"], sp_weights=None, combiner="mean")
            actor_embedding = tf.nn.embedding_lookup_sparse(actor_look_table, sp_ids=features["actor"], sp_weights=None, combiner="mean")
            director_embedding = tf.nn.embedding_lookup_sparse(director_look_table, sp_ids=features["director"], sp_weights=None, combiner="mean")

            # _embedding = tf.nn.embedding_lookup(_lookup_table,
            #                                     tf.string_to_hash_bucket_fast(features[""],
            #                                                                   feat_config["n_"]))
            #
            # _embedding = tf.nn.embedding_lookup(_lookup_table,
            #                                     tf.string_to_hash_bucket_fast(features[""],
            #                                                                   feat_config["n_"]))
            # item sequence

            seq_item_embedding = tf.nn.embedding_lookup(item_lookup_table,
                                                        tf.string_to_hash_bucket_fast(features["seq_item"],
                                                                                      feat_config["n_item"]))
            seq_rating_embedding = tf.nn.embedding_lookup(rating_lookup_table,
                                                       tf.string_to_hash_bucket_fast(features["seq_rating"],
                                                                                     feat_config["n_rating"]))
            # gift sequence
            # ia
            gift_ia_item_embedding = tf.nn.embedding_lookup(item_lookup_table,
                                                            tf.string_to_hash_bucket_fast(features["gift_ia_item"],
                                                                                          feat_config["n_item"]))
            gift_ia_rating_embedding = tf.nn.embedding_lookup(rating_lookup_table,
                                                              tf.string_to_hash_bucket_fast(features["gift_ia_rating"],
                                                                                            feat_config["n_rating"]))
            # id
            gift_id_item_embedding = tf.nn.embedding_lookup(item_lookup_table,
                                                            tf.string_to_hash_bucket_fast(features["gift_id_item"],
                                                                                          feat_config["n_item"]))
            gift_id_rating_embedding = tf.nn.embedding_lookup(rating_lookup_table,
                                                            tf.string_to_hash_bucket_fast(features["gift_id_rating"],
                                                                                          feat_config["n_rating"]))
            # gift_actor_embedding = tf.nn.embedding_lookup(actor_lookup_table,
            #                                                tf.string_to_hash_bucket_fast(features["gift_ia_author"],
            #                                                                              feat_config["n_author"]))
            #
            # gift_publisher_embedding = tf.nn.embedding_lookup(publisher_lookup_table,
            #                                                   tf.string_to_hash_bucket_fast(features["gift_publisher"],
            #                                                                                 feat_config["n_publisher"]))

            # concatenate the tensors
            tensor_dict = {}
            tensor_dict["user_embedding"] = tf.concat([user_embedding, age_embedding, zip_embedding], 1)
            tensor_dict["item_embedding"] = tf.concat([item_embedding, rating_embedding, genre_embedding, actor_embedding, director_embedding], 1)
            tensor_dict["opt_seq_embedding"] = tf.concat([seq_item_embedding, seq_rating_embedding], 2)

            # gift feature
            # ia
            tensor_dict["gift_ia_embedding"] = tf.concat([gift_ia_item_embedding, gift_ia_rating_embedding], 2)
            tensor_dict["gift_ia_length"] = features["gift_ia_length"]
            tensor_dict["label"] = features["label"]

            # id
            tensor_dict["gift_id_embedding"] = tf.concat([gift_id_item_embedding, gift_id_rating_embedding], 2)
            tensor_dict["gift_id_length"] = features["gift_id_length"]
            tensor_dict["label"] = features["label"]

        return tensor_dict

if __name__ == '__main__':

    file = "../FeatGeneration/MovieLens-1M/ui_sample_gift_new_test.csv"

    fg = FeatGenerator(file)
    features = fg.feature_generation()

    tg = TensorGenerator()
    tensor_dict = tg.embedding_layer(features, fg.feat_config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("pause")
    # sess.run(train_tensor_dict["user_embedding"])

