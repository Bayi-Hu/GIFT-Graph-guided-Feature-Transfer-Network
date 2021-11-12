import numpy
import json
import pickle as pkl
import random
import gzip
import shuffle

if __name__ == '__main__':
    test_file = "./Amazon/local_test_splitByUser"
    test_file_new = "./Amazon/local_test_splitByUser_new"

    f_read = open(test_file, "r")
    f_write = open(test_file_new, "w")

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

        arr_new = "\t".join([arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], str(length_iid)]) + "\n"

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


