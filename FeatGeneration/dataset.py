import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


test_file = "/Amazon/local_test_splitByUser"

f = open(test_file, 'rb')

def parse_split(line):
    parse_res = tf.string_split([line], delimiter="\t")
    values = parse_res.values
    label = values[0]
    iid_sequence = values[4]
    cat_sequence = values[5]
    return label, iid_sequence, cat_sequence

# ---
dataset = tf.data.TextLineDataset(test_file)
dataset = dataset.map(parse_split, num_parallel_calls=2)
dataset = dataset.shuffle(3).repeat(1).batch(128)


iterator = dataset.make_one_shot_iterator()
# 这里get_next()返回一个字符串类型的张量，代表文件中的一行。
label, iid_sequence, cat_sequnece = iterator.get_next()

sess = tf.Session()
print("pause for debug...")


# with tf.Session() as sess:
#     for i in range(10):
#         print(sess.run(x))

