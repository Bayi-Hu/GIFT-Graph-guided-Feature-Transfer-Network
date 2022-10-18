#-*- coding:utf-8 -*-
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from FeatGeneration.fg_DBook import FeatGenerator, TensorGenerator
from Model.DNN import DNN

if __name__ == '__main__':

    train_file = "../../FeatGeneration/DBook/ui_sample_gift_full_train.csv"

    train_fg = FeatGenerator(train_file)
    train_features = train_fg.feature_generation()
    tg = TensorGenerator()
    train_tensor_dict = tg.embedding_layer(train_features, train_fg.feat_config)

    model = DNN(train_tensor_dict, train_config={"is_training": True, "dropout_rate": 0.2})
    model.build()

    checkpoint_dir = "./save_log"
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iter = 0
        while True:
        # for i in range(100):
            try:
                _, loss, acc = sess.run([model.optimizer, model.loss, model.accuracy])

                if iter % 100 == 0:
                    print("iter=%d, loss=%f, acc=%f" %(iter, loss, acc))

                iter += 1
            except Exception as e:
                print(e)
                # save model
                saver.save(sess, os.path.join(checkpoint_dir, "model_1"))
                break


