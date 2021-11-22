import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

csv = [
  "1,harden|james|curry",
  "2,wrestbrook|harden|durant",
  "3,|paul|towns",
]

TAG_SET = ["harden", "james", "curry", "durant", "paul","towns","wrestbrook"]

def sparse_from_csv(csv):
  ids, post_tags_str = tf.decode_csv(csv, [[-1], [""]])
  init = tf.lookup.KeyValueTensorInitializer(TAG_SET, tf.constant(np.arange(0, len(TAG_SET))))
  table = tf.lookup.StaticHashTable(init, default_value=-1)
  split_tags = tf.string_split(post_tags_str, "|")
  return tf.SparseTensor(
          indices=split_tags.indices,
          values=table.lookup(split_tags.values), ## 这里给出了不同值通过表查到的index ##
          dense_shape=split_tags.dense_shape)


TAG_EMBEDDING_DIM = 3
embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET), TAG_EMBEDDING_DIM]))
tags = sparse_from_csv(csv)
embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)

with tf.Session() as s:
  s.run([tf.global_variables_initializer(), tf.tables_initializer()])
  print(s.run([embedded_tags]))
  # sess.run(tf.tables_initializer())
