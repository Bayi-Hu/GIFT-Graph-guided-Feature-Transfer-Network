import numpy
import json
import pickle as pkl
import random
import gzip
import shuffle

if __name__ == '__main__':

    test_file = "./Amazon/local_test_splitByUser"
    # train_file = "./Amazon/local_train_splitByUser"
    test_file_new = "./Amazon/local_test_splitByUser_new"
    # train_file_new = "./Amazon/local_train_splitByUser_new"

    # f_read = open(train_file, "r")
    # f_write = open(train_file_new, "w")

    f_read = open(test_file, "r")
    f_write = open(test_file_new, "w")

    maxlen = 100

    for line in f_read:
        arr = line.strip().split("\t")
        label = arr[0]
        user_id = arr[1]
        item_id = arr[2]
        category = arr[3]
        seq_item_id = arr[4].split("")
        seq_category = arr[5].split("")

        length_iid = len(seq_item_id)
        length_category = len(seq_category)
        assert length_iid == length_category

        if length_iid >= 100:
            seq_item_id = seq_item_id[:100]
            seq_category = seq_category[:100]
            length_iid = 100

        arr_new = "\t".join([label, user_id, item_id, category, "".join(seq_item_id), "".join(seq_category), str(length_iid)]) + "\n"
        f_write.writelines(arr_new)

def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            # return unicode_to_utf8(json.load(f))
            return json.load(f)
    except:
        with open(filename, 'rb') as f:
            # return unicode_to_utf8(pkl.load(f))
            return pkl.load(f)


