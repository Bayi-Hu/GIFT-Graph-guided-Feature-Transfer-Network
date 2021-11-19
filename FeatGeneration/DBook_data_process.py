#-*- coding:utf-8 -*-
import pandas as pd
import collections
import numpy as np
import re

def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

# --------------------------

input_dir = 'DBook/'

# read logs
ui_data = pd.read_csv(input_dir+'user_book.txt', names=['user','item','rating'], sep='\t',engine='python')
ui_data.loc[ui_data[ui_data.rating < 4].index, 'label'] = "0"
ui_data.loc[ui_data[ui_data.rating >= 4].index, 'label'] = "1"
ui_data = ui_data.astype("str")
pos_ui_data = ui_data[ui_data.label==1].copy()

# read item/user feature
ul = pd.read_csv(input_dir+'user_location.txt', names=['user','location'], sep='\t',engine='python')
ug = pd.read_csv(input_dir+'user_group.txt', names=['user','group'], sep='\t', engine='python')

ba = pd.read_csv(input_dir+'book_author.txt', names=['book','author'], sep='\t',engine='python')
bp = pd.read_csv(input_dir+'book_publisher.txt', names=['book','publisher'], sep='\t',engine='python')
by = pd.read_csv(input_dir+'book_year.txt', names=['book','year'], sep='\t',engine='python')

user_list = list(set(ui_data.user) & set(ul.user))
item_list = list(set(ui_data.item) & ((set(ba.book) & set(bp.book))) & set(by.book))

print(len(user_list), len(item_list))

location_list = list(set(ul[ul.user.isin(user_list)].location))
group_list = list(set(ug[ug.user.isin(user_list)].location))
publisher_list = list(set(bp[bp.book.isin(item_list)].publisher))
author_list = list(set(ba[ba.book.isin(item_list)].author))
len(location_list), len(group_list), len(publisher_list), len(author_list)

# user feature

def udf(df):
    return np.array([[df.iloc[0]["user"], "".join(df.group.values.astype(str))]])

ug_ = ug.groupby(["user"]).apply(udf)
ugs = pd.DataFrame(np.concatenate(ug_.values,axis=0),columns=["user","groups"])

user_feat_df = pd.merge(left=ugs, right=ul, on="user", how="inner")
item_feat_df = pd.merge(left=pd.merge(left=bp, right=ba, how="inner"), right=by, on="user", how="inner")

# training_sample 不构建复杂的sequence特征, 因为没有timestamp信息
