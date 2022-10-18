#-*- coding:utf-8 -*-
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from FeatGeneration.fg_DBook import FeatGenerator, TensorGenerator
from Model.DNN import DNN
from sklearn import metrics
import numpy as np

if __name__ == '__main__':

    test_file = "../../FeatGeneration/DBook/ui_sample_gift_new_test.csv"

    test_fg = FeatGenerator(test_file)
    test_features = test_fg.feature_generation()
    tg = TensorGenerator()
    test_tensor_dict = tg.embedding_layer(test_features, test_fg.feat_config)

    model = DNN(test_tensor_dict, train_config={"is_training": False, "dropout_rate": 0})
    model.build()

    ckpt = "./save_log/model_1"
    saver = tf.train.Saver()

    logits = []
    labels = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        iter = 0
        while True:
        # for i in range(10):
            try:
                logit, label, loss, acc = sess.run([model.y_hat, model.label, model.loss, model.accuracy])

                logits.append(logit)
                labels.append(label)

                if iter % 100 == 0:
                    print("iter=%d, loss=%f, acc=%f" %(iter, loss, acc))

                iter += 1
            except Exception as e:
                print(e)
                # save model
                # saver.save(sess, os.path.join(checkpoint_dir, "model_"+str(iter)))
                break

    print("pause")
    labels = np.concatenate(labels, axis=0)[:,1]
    logits = np.concatenate(logits, axis=0)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_true=labels, y_score=logits, pos_label=1)
    auc_value = metrics.auc(fpr, tpr)
    print("auc:", auc_value)
    print("pause")





