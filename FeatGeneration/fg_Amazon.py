import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


feat_config = {
    "n_uid": 1000,
    "d_uid": 32,
    "n_iid": 1000,
    "d_iid": 32,
    "n_cid": 1000,
    "d_cid": 32,
    "max_length": 50
}


def parse_split(line):
    parse_res = tf.string_split([line], delimiter="\t")
    values = parse_res.values
    label = values[0]
    user_id = values[1]
    item_id = values[2]
    category = values[3]
    iid_sequence = values[4]
    cat_sequence = values[5]
    # length = values[6]
    # length = tf.string_split([iid_sequence], delimiter="")
    return label, user_id, item_id, category, iid_sequence, cat_sequence
        # , length

# ---

def feature_generation(input_file):
    """
    Args:
        input_file: a .txt file that includes the training or testing sample
    Returns:
        feature tensor used for training or testing
    """
    dataset = tf.data.TextLineDataset(input_file)
    dataset = dataset.map(parse_split, num_parallel_calls=2)
    dataset = dataset.shuffle(3).repeat(1).batch(128)
    iterator = dataset.make_one_shot_iterator()

    label, user_id, item_id, category, seq_item_id, seq_category = iterator.get_next()
        # , length = iterator.get_next()

    features = {}
    features["label"] = label
    features["user_id"] = user_id
    features["item_id"] = item_id
    features["category"] = category
    features["seq_item_id"] = seq_item_id
    features["seq_category"] = seq_category
    # features["length"] = length

    return features

# if __name__ == '__main__':
#
#     test_file = "./Amazon/local_test_splitByUser"
#     features = feature_generation(test_file)
#     sess = tf.Session()

def embedding_layer(features):

    with tf.name_scope('Embedding_layer'):

        uid_lookup_table = tf.get_variable("uid_embedding_var", [feat_config["n_uid"], feat_config["d_uid"]])
        iid_lookup_table = tf.get_variable("iid_embedding_var", [feat_config["n_iid"], feat_config["d_iid"]])
        cat_lookup_table = tf.get_variable("cat_embedding_var", [feat_config["n_cid"], feat_config["d_cid"]])

        # add to summary
        # tf.summary.histogram('uid_lookup_table', uid_lookup_table)
        # tf.summary.histogram('iid_lookup_table', iid_lookup_table)
        # tf.summary.histogram('cat_lookup_table', cat_lookup_table)

        uid_embedding = tf.nn.embedding_lookup(uid_lookup_table, features["user_id"])
        iid_embedding = tf.nn.embedding_lookup(iid_lookup_table, features["item_id"])
        cat_embedding = tf.nn.embedding_lookup(cat_lookup_table, features["category"])

        # item sequence
        seq_iid_embedding = tf.nn.embedding_lookup(iid_lookup_table, features["seq_item_id"])
        seq_cat_embedding = tf.nn.embedding_lookup(cat_lookup_table, features["seq_category"])

        # concatenate the tensors
        tensor_dict = {}
        tensor_dict["user_embedding"] = uid_embedding
        tensor_dict["item_embedding"] = tf.concat([iid_embedding, cat_embedding], 1)
        tensor_dict["opt_seq_embedding"] = tf.concat([seq_iid_embedding, seq_cat_embedding], 2)
        tensor_dict["length"] = features["length"]
        tensor_dict["label"] = features["label"]

    return tensor_dict


