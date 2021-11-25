# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

SEQ_MAX_LENGTH = 50
GIFT_IA_MAX_LENGTH = 30
GIFT_ID_MAX_LENGTH = 30

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
            "epoch": 1
        }

    def parse_split(self, line):
        parse_res = tf.string_split([line], delimiter="\t")
        values = parse_res.values

        user = values[0]
        item = values[1]
        # timestamp = values[2]
        label = values[3]

        length = values[4]
        seq_item = values[5]
        seq_rating = values[6]
        seq_genre = values[7]
        seq_director = values[8]
        seq_actor = values[9]

        gender = values[10]
        age = values[11]
        occupation = values[12]
        zip = values[13]
        rating = values[14]

        genre = values[15]
        director = values[16]
        actor = values[17]

        # gift feature
        gift_ia_item = values[18]
        gift_ia_rating = values[19]
        gift_ia_genre = values[20]
        gift_ia_director = values[21]
        gift_ia_actor = values[22]
        gift_ia_length = values[23]

        gift_id_item = values[24]
        gift_id_rating = values[25]
        gift_id_genre = values[26]
        gift_id_director = values[27]
        gift_id_actor = values[28]
        gift_id_length = values[29]

        return label, user, item, gender, age, occupation, zip, rating, genre, director, actor, seq_item, seq_rating, seq_genre, seq_actor, seq_director, length, \
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

        label, user, item, gender, age, occupation, zip, rating, genre, director, actor, seq_item, seq_rating, seq_genre, seq_actor, seq_director, length, \
        gift_ia_item, gift_ia_rating, gift_ia_genre, gift_ia_director, gift_ia_actor, gift_ia_length, \
        gift_id_item, gift_id_rating, gift_id_genre, gift_id_director, gift_id_actor, gift_id_length = iterator.get_next()

        #
        def nan_convert(x):
            if tf.equal(x, "nan"):
                return "0"
            else:
                return x


        genre = tf.string_split(genre, delimiter="")
        genre = tf.SparseTensor(indices=genre.indices,
                                values=tf.string_to_number(tf.map_fn(nan_convert, genre.values), out_type=tf.int32),
                                dense_shape=genre.dense_shape) # convert string to int32

        actor = tf.string_split(actor, delimiter="")
        actor = tf.SparseTensor(indices=actor.indices,
                                values=tf.string_to_number(tf.map_fn(nan_convert, actor.values), out_type=tf.int32),
                                dense_shape=actor.dense_shape)  # convert string to int32

        director = tf.string_split(director, delimiter="")
        director = tf.SparseTensor(indices=director.indices,
                                   values=tf.string_to_number(tf.map_fn(nan_convert, director.values), out_type=tf.int32),
                                   dense_shape=director.dense_shape)  #

        # sequence
        seq_item = self.parse_sequence(seq_item, max_length=SEQ_MAX_LENGTH, delimiter=",", default_value="nan")
        seq_rating = self.parse_sequence(seq_rating, max_length=SEQ_MAX_LENGTH, delimiter=",", default_value="nan")
        seq_genre = self.parse_sequence(seq_genre, max_length=SEQ_MAX_LENGTH, delimiter=",", default_value="nan")
        seq_actor = self.parse_sequence(seq_actor, max_length=SEQ_MAX_LENGTH, delimiter=",", default_value="nan")
        seq_director = self.parse_sequence(seq_director, max_length=SEQ_MAX_LENGTH, delimiter=",", default_value="nan")

        print("pause")
        seq_genre_split = tf.string_split(tf.reshape(seq_genre, shape=[-1]), delimiter="")
        seq_genre = tf.SparseTensor(indices=seq_genre_split.indices,
                                    values=tf.string_to_number(tf.map_fn(nan_convert, seq_genre_split.values), out_type=tf.int32),
                                    dense_shape=seq_genre_split.dense_shape)  # convert string to int32

        seq_actor_split = tf.string_split(tf.reshape(seq_actor, shape=[-1]), delimiter="")
        seq_actor = tf.SparseTensor(indices=seq_actor_split.indices,
                                    values=tf.string_to_number(tf.map_fn(nan_convert, seq_actor_split.values), out_type=tf.int32),
                                    dense_shape=seq_actor_split.dense_shape)  # convert string to int32

        seq_director_split = tf.string_split(tf.reshape(seq_director, shape=[-1]), delimiter="")
        seq_director = tf.SparseTensor(indices=seq_director_split.indices,
                                       values=tf.string_to_number(tf.map_fn(nan_convert, seq_director_split.values), out_type=tf.int32),
                                       dense_shape=seq_director_split.dense_shape)  # convert string to int32


        # gift sequence
        # ia
        gift_ia_item = self.parse_sequence(gift_ia_item, max_length=GIFT_IA_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_ia_rating = self.parse_sequence(gift_ia_rating, max_length=GIFT_IA_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_ia_genre = self.parse_sequence(gift_ia_genre, max_length=GIFT_IA_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_ia_actor = self.parse_sequence(gift_ia_actor, max_length=GIFT_IA_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_ia_director = self.parse_sequence(gift_ia_director, max_length=GIFT_IA_MAX_LENGTH, delimiter=",", default_value="nan")

        gift_ia_genre_split = tf.string_split(tf.reshape(gift_ia_genre, shape=[-1]), delimiter="")
        gift_ia_genre = tf.SparseTensor(indices=gift_ia_genre_split.indices,
                                        values=tf.string_to_number(tf.map_fn(nan_convert, gift_ia_genre_split.values), out_type=tf.int32),
                                        dense_shape=gift_ia_genre_split.dense_shape)  # convert string to int32

        gift_ia_actor_split = tf.string_split(tf.reshape(gift_ia_actor, shape=[-1]), delimiter="")
        gift_ia_actor = tf.SparseTensor(indices=gift_ia_actor_split.indices,
                                        values=tf.string_to_number(tf.map_fn(nan_convert, gift_ia_actor_split.values), out_type=tf.int32),
                                        dense_shape=gift_ia_actor_split.dense_shape)  # convert string to int32

        gift_ia_director_split = tf.string_split(tf.reshape(gift_ia_director, shape=[-1]), delimiter="")
        gift_ia_director = tf.SparseTensor(indices=gift_ia_director_split.indices,
                                           values=tf.string_to_number(tf.map_fn(nan_convert, gift_ia_director_split.values), out_type=tf.int32),
                                           dense_shape=gift_ia_director_split.dense_shape)  # convert string to int32

        # id

        gift_id_item = self.parse_sequence(gift_id_item, max_length=GIFT_ID_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_id_rating = self.parse_sequence(gift_id_rating, max_length=GIFT_ID_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_id_genre = self.parse_sequence(gift_id_genre, max_length=GIFT_ID_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_id_actor = self.parse_sequence(gift_id_actor, max_length=GIFT_ID_MAX_LENGTH, delimiter=",", default_value="nan")
        gift_id_director = self.parse_sequence(gift_id_director, max_length=GIFT_ID_MAX_LENGTH, delimiter=",", default_value="nan")

        # multihot

        gift_id_genre_split = tf.string_split(tf.reshape(gift_id_genre, shape=[-1]), delimiter="")
        gift_id_genre = tf.SparseTensor(indices=gift_id_genre_split.indices,
                                        values=tf.string_to_number(tf.map_fn(nan_convert, gift_id_genre_split.values), out_type=tf.int32),
                                        dense_shape=gift_id_genre_split.dense_shape)  # convert string to int32

        gift_id_actor_split = tf.string_split(tf.reshape(gift_id_actor, shape=[-1]), delimiter="")
        gift_id_actor = tf.SparseTensor(indices=gift_id_actor_split.indices,
                                        values=tf.string_to_number(tf.map_fn(nan_convert, gift_id_actor_split.values), out_type=tf.int32),
                                        dense_shape=gift_id_actor_split.dense_shape)  # convert string to int32

        gift_id_director_split = tf.string_split(tf.reshape(gift_id_director, shape=[-1]), delimiter="")
        gift_id_director = tf.SparseTensor(indices=gift_id_director_split.indices,
                                           values=tf.string_to_number(tf.map_fn(nan_convert, gift_id_director_split.values), out_type=tf.int32),
                                           dense_shape=gift_id_director_split.dense_shape)  # convert string to int32

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
        features["seq_actor"] = seq_actor
        features["seq_director"] = seq_director
        features["length"] = tf.string_to_number(length, out_type=tf.int32)

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
            genre_look_table = tf.get_variable("genre_embedding_var", [feat_config["n_genre"], feat_config["d_genre"]])
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
            occupation_embedding = tf.nn.embedding_lookup(occupation_lookup_table,
                                                          tf.string_to_hash_bucket_fast(features["occupation"],
                                                                                        feat_config["n_occupation"]))

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

            # opt_sequence
            seq_item_embedding = tf.nn.embedding_lookup(item_lookup_table,
                                                        tf.string_to_hash_bucket_fast(features["seq_item"],
                                                                                      feat_config["n_item"]))
            seq_rating_embedding = tf.nn.embedding_lookup(rating_lookup_table,
                                                       tf.string_to_hash_bucket_fast(features["seq_rating"],
                                                                                     feat_config["n_rating"]))
            seq_genre_embedding = tf.nn.embedding_lookup_sparse(genre_look_table, sp_ids=features["seq_genre"],
                                                                sp_weights=None, combiner="mean")
            seq_genre_embedding = tf.reshape(seq_genre_embedding, [-1, SEQ_MAX_LENGTH, feat_config["d_genre"]])

            seq_actor_embedding = tf.nn.embedding_lookup_sparse(actor_look_table, sp_ids=features["seq_actor"],
                                                                sp_weights=None, combiner="mean")

            seq_actor_embedding = tf.reshape(seq_actor_embedding, [-1, SEQ_MAX_LENGTH, feat_config["d_actor"]])

            seq_director_embedding = tf.nn.embedding_lookup_sparse(actor_look_table, sp_ids=features["seq_director"],
                                                                   sp_weights=None, combiner="mean")
            seq_director_embedding = tf.reshape(seq_director_embedding, [-1, SEQ_MAX_LENGTH, feat_config["d_director"]])

            # gift sequence
            # ia
            gift_ia_item_embedding = tf.nn.embedding_lookup(item_lookup_table,
                                                            tf.string_to_hash_bucket_fast(features["gift_ia_item"],
                                                                                          feat_config["n_item"]))
            gift_ia_rating_embedding = tf.nn.embedding_lookup(rating_lookup_table,
                                                              tf.string_to_hash_bucket_fast(features["gift_ia_rating"],
                                                                                            feat_config["n_rating"]))

            gift_ia_genre_embedding = tf.nn.embedding_lookup_sparse(genre_look_table, sp_ids=features["gift_ia_genre"],
                                                                    sp_weights=None, combiner="mean")
            gift_ia_genre_embedding = tf.reshape(gift_ia_genre_embedding, [-1, GIFT_IA_MAX_LENGTH, feat_config["d_genre"]])

            gift_ia_actor_embedding = tf.nn.embedding_lookup_sparse(actor_look_table, sp_ids=features["gift_ia_actor"],
                                                                    sp_weights=None, combiner="mean")

            gift_ia_actor_embedding = tf.reshape(gift_ia_actor_embedding, [-1, GIFT_IA_MAX_LENGTH, feat_config["d_actor"]])

            gift_ia_director_embedding = tf.nn.embedding_lookup_sparse(director_look_table,
                                                                       sp_ids=features["gift_ia_director"],
                                                                       sp_weights=None, combiner="mean")
            gift_ia_director_embedding = tf.reshape(gift_ia_director_embedding, [-1, GIFT_IA_MAX_LENGTH, feat_config["d_director"]])

            # id
            gift_id_item_embedding = tf.nn.embedding_lookup(item_lookup_table,
                                                            tf.string_to_hash_bucket_fast(features["gift_id_item"],
                                                                                          feat_config["n_item"]))

            gift_id_rating_embedding = tf.nn.embedding_lookup(rating_lookup_table,
                                                              tf.string_to_hash_bucket_fast(features["gift_id_rating"],
                                                                                            feat_config["n_rating"]))

            gift_id_genre_embedding = tf.nn.embedding_lookup_sparse(genre_look_table, sp_ids=features["gift_id_genre"],
                                                                    sp_weights=None, combiner="mean")

            gift_id_genre_embedding = tf.reshape(gift_id_genre_embedding, [-1, GIFT_ID_MAX_LENGTH, feat_config["d_genre"]])

            gift_id_actor_embedding = tf.nn.embedding_lookup_sparse(actor_look_table, sp_ids=features["gift_id_actor"],
                                                                    sp_weights=None, combiner="mean")

            gift_id_actor_embedding = tf.reshape(gift_id_actor_embedding, [-1, GIFT_ID_MAX_LENGTH, feat_config["d_actor"]])

            gift_id_director_embedding = tf.nn.embedding_lookup_sparse(director_look_table, sp_ids=features["gift_id_director"],
                                                                       sp_weights=None, combiner="mean")
            gift_id_director_embedding = tf.reshape(gift_id_director_embedding, [-1, GIFT_ID_MAX_LENGTH, feat_config["d_director"]])

            # concatenate the tensors
            tensor_dict = {}
            tensor_dict["user_embedding"] = tf.concat([user_embedding, age_embedding, occupation_embedding, zip_embedding], 1)
            tensor_dict["item_embedding"] = tf.concat([item_embedding, rating_embedding, genre_embedding, actor_embedding, director_embedding], 1)
            tensor_dict["opt_seq_embedding"] = tf.concat([seq_item_embedding, seq_rating_embedding, seq_genre_embedding, seq_actor_embedding, seq_director_embedding], 2)
            tensor_dict["length"] = features["length"]

            # gift feature
            # ia
            tensor_dict["gift_ia_embedding"] = tf.concat([gift_ia_item_embedding, gift_ia_rating_embedding, gift_ia_genre_embedding, gift_ia_actor_embedding, gift_ia_director_embedding], 2)
            tensor_dict["gift_ia_length"] = features["gift_ia_length"]
            tensor_dict["label"] = features["label"]

            # id
            tensor_dict["gift_id_embedding"] = tf.concat([gift_id_item_embedding, gift_id_rating_embedding, gift_id_genre_embedding, gift_id_actor_embedding, gift_id_director_embedding], 2)
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

