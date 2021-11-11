#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from FeatGeneration.fg_Amazon import FeatGenerator, TensorGenerator
from Model.model import Model

if __name__ == '__main__':

    train_file = "../../FeatGeneration/Amazon/local_train_splitByUser"
    test_file = "../../FeatGeneration/Amazon/local_test_splitByUser"

    train_fg = FeatGenerator(train_file)
    train_features = train_fg.feature_generation()
    tg = TensorGenerator()
    train_tensor_dict = tg.embedding_layer(train_features, train_fg.feat_config)

    # test_fg = FeatGenerator(test_file)
    # test_features = test_fg.feature_generation()
    # test_tensor_dict = tg.embedding_layer(test_features, test_fg.feat_config)

    model = Model(train_tensor_dict)
    model.build()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        for i in range(100):
            try:
                _, loss, acc = sess.run([model.optimizer, model.loss, model.accuracy])
                print("loss=", loss)
                print("acc=", acc)
            except:
                break


