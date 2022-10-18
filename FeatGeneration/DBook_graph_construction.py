#-*- coding:utf-8 -*-
import pandas as pd
import igraph
import collections
import numpy as np
import re

input_dir = 'DBook/'

# read item feature
ba = pd.read_csv(input_dir+'book_author.txt', names=['book','author'], sep='\t',engine='python').astype(str)
bp = pd.read_csv(input_dir+'book_publisher.txt', names=['book','publisher'], sep='\t',engine='python').astype(str)
by = pd.read_csv(input_dir+'book_year.txt', names=['book','year'], sep='\t',engine='python').astype(str)

item_feat_df = pd.merge(left=pd.merge(left=bp, right=ba, how="inner"), right=by, on="book", how="inner")

item_id_list = item_feat_df.book.values
author_id_list = item_feat_df.author.values
publisher_id_list = item_feat_df.publisher.values
year_list = item_feat_df.year.values

print("pause")

def item_suffix(x):
    return "i_"+x

def actor_suffix(x):
    return "a_"+x

def director_suffix(x):
    return "d_"+x

# 需要将rate_id_list 等转为item_id2rate_id, 因为item_id_list和index并不是一一对应

iid2author_id = dict(zip(item_id_list, author_id_list))
iid2publisher_id = dict(zip(item_id_list, publisher_id_list))
iid2year = dict(zip(item_id_list, year_list))

# build graph
i_vertices = list(map(item_suffix, item_id_list.astype(str)))
a_vertices = list(map(actor_suffix, author_id_list.astype(str)))

# add edges
ia_edges = []
for idx, row in item_feat_df[["book","author"]].iterrows():
    ia_edges.append(("i_"+row.book, "a_"+row.author))


g_ia = igraph.Graph()
g_ia.add_vertices(i_vertices)
g_ia.add_vertices(a_vertices)
g_ia.add_edges(ia_edges)

print("pause")

idx2string_ia = dict(zip(range(len(g_ia.vs["name"])), g_ia.vs["name"]))
string2idx_ia = dict(zip(g_ia.vs["name"], range(len(g_ia.vs["name"]))))
MAX_LENGTH_IA = 50

def mapGid2Iid_ia(x):
    # x is a list of lists
    y = idx2string_ia[x]
    y = y.split("_")[1]
    return y

gift_ia_iids = []
gift_ia_author = []
gift_ia_publisher = []
gift_ia_year = []
gift_ia_length = []

for iid in i_vertices:

    gid = string2idx_ia[iid] # use for reduce itself
    g_aids = g_ia.neighbors(vertex = iid)
    g_iids = g_ia.neighborhood(vertices=g_aids)
    g_iids = np.concatenate(g_iids, axis=0)
    g_iids = list(set(g_iids).difference(set(g_aids)).difference([gid]))
    iids = list(map(mapGid2Iid_ia, g_iids))
    iids = np.random.permutation(iids)

    print(len(iids))

    if len(iids)> MAX_LENGTH_IA:
        iids = iids[:MAX_LENGTH_IA]
        num = MAX_LENGTH_IA
    else:
        num = len(iids)

    # 将特征进行组合, 注意需要将sequence特征拆开
    # feature 有 rating, actors, directors,
    author_seq = []
    publisher_seq = []
    year_seq = []

    for iid in iids:

        author_seq.append(iid2author_id[iid])
        publisher_seq.append(iid2publisher_id[iid])
        year_seq.append(iid2year[iid])

    iids = ",".join(iids)
    author_seq = ",".join(author_seq)
    publisher_seq = ",".join(publisher_seq)
    year_seq = ",".join(year_seq)

    gift_ia_iids.append(iids)
    gift_ia_author.append(author_seq)
    gift_ia_publisher.append(publisher_seq)
    gift_ia_year.append(year_seq)
    gift_ia_length.append(num)

# generate dataframe for item feat
gift_ia_df = pd.DataFrame({
    "item": item_id_list,
    "gift_ia_item": gift_ia_iids,
    "gift_ia_author": gift_ia_author,
    "gift_ia_publisher": gift_ia_publisher,
    "gift_ia_year": gift_ia_year,
    "gift_ia_length": gift_ia_length
})

# store
gift_ia_df.to_csv(input_dir+"gift_df.csv", index=False)
