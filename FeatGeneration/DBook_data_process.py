#-*- coding:utf-8 -*-
import pandas as pd
import collections
import numpy as np
import re

input_dir = 'DBook/'

# read logs
ui_data = pd.read_csv(input_dir+'user_book.txt', names=['user','item','rating'], sep='\t',engine='python')
ui_data.loc[ui_data[ui_data.rating < 5].index, 'label'] = "0"
ui_data.loc[ui_data[ui_data.rating >= 5].index, 'label'] = "1"
ui_data = ui_data.astype(str)
pos_ui_data = ui_data[ui_data.label==1].copy()

# read item/user feature
ul = pd.read_csv(input_dir+'user_location.txt', names=['user','location'], sep='\t',engine='python').astype(str)
ug = pd.read_csv(input_dir+'user_group.txt', names=['user','group'], sep='\t', engine='python').astype(str)

ba = pd.read_csv(input_dir+'book_author.txt', names=['book','author'], sep='\t',engine='python').astype(str)
bp = pd.read_csv(input_dir+'book_publisher.txt', names=['book','publisher'], sep='\t',engine='python').astype(str)
by = pd.read_csv(input_dir+'book_year.txt', names=['book','year'], sep='\t',engine='python').astype(str)

user_list = list(set(ui_data.user) & set(ul.user))
item_list = list(set(ui_data.item) & ((set(ba.book) & set(bp.book))) & set(by.book))

print(len(user_list), len(item_list))

location_list = list(set(ul[ul.user.isin(user_list)].location))
group_list = list(set(ug[ug.user.isin(user_list)].group))
publisher_list = list(set(bp[bp.book.isin(item_list)].publisher))
author_list = list(set(ba[ba.book.isin(item_list)].author))
print(len(location_list), len(group_list), len(publisher_list), len(author_list))

# user feature

def udf(df):
    return np.array([[df.iloc[0]["user"], "".join(df.group.values.astype(str))]])

ug_ = ug.groupby(["user"]).apply(udf)
ugs = pd.DataFrame(np.concatenate(ug_.values,axis=0),columns=["user","groups"])

user_feat_df = pd.merge(left=ugs, right=ul.astype(str), on="user", how="inner")
item_feat_df = pd.merge(left=pd.merge(left=bp.astype(str), right=ba.astype(str), how="inner"), right=by.astype(str), on="book", how="inner")

item_feat_df.rename(columns={"book":"item"}, inplace=True) # change book to item

#---
print("pause")
gift_feat = pd.read_csv(input_dir+"gift_df.csv").astype(str)
ui_sample_base = pd.merge(left=pd.merge(left=ui_data[["user","item","label"]], right=user_feat_df, on="user", how="inner"),right=item_feat_df, on="item", how="inner")
ui_sample_gift = pd.merge(left=ui_sample_base, right=gift_feat, on="item", how="left")
ui_sample_gift = ui_sample_gift.sample(frac=1)
# ui_sample_gift.to_csv(input_dir+"ui_sample_gift.csv", sep="\t", header=0, index=0, na_rep="")

# split old and new movies: rules: divided the movies into movies released before 1997 and after 1998 (approximately 8:2)
# new: >= 40
ui_sample_gift_new = ui_sample_gift[ui_sample_gift.year.astype(int)>=40].copy()
ui_sample_gift_old = ui_sample_gift[ui_sample_gift.year.astype(int)<40].copy()

ui_sample_gift_old = ui_sample_gift_old.sample(frac=1)
# make the new item data more sparse, otherwise it is not the "real" new item.
ui_sample_gift_new = ui_sample_gift_new.sample(frac=0.5)

# shuffle
ui_sample_gift_old = ui_sample_gift_old.sample(frac=1)
ui_sample_gift_new = ui_sample_gift_new.sample(frac=1)

# write in data
ui_sample_gift_new.to_csv(input_dir+"ui_sample_gift_new_test.csv", sep="\t", header=0, index=0, na_rep="")
ui_sample_gift_old.to_csv(input_dir+"ui_sample_gift_full_train.csv", sep="\t", header=0, index=0, na_rep="")

