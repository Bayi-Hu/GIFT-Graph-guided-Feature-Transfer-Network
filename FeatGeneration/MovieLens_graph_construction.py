#-*- coding:utf-8 -*-
import pandas as pd
import collections
import numpy as np
import igraph
import re

def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

# --------------------------

input_dir = 'MovieLens-1M/'

# read logs
ui_data = pd.read_csv(input_dir+'ratings.txt', names=['user', 'item', 'rating', 'timestamp'],sep="::", engine='python')

ui_data.loc[ui_data[ui_data.rating < 4].index, 'label'] = "0"
ui_data.loc[ui_data[ui_data.rating >= 4].index, 'label'] = "1"

ui_data = ui_data.astype("string")

print(len(ui_data))

# read item feature
item_data = pd.read_csv(input_dir+'movies_extrainfos.txt', names=['item', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'],
                        sep="::", engine='python', encoding="utf-8")

item_list = list(set(ui_data.item.tolist()) | set(item_data.item))

rate_list = load_list("{}/m_rate.txt".format(input_dir))
genre_list = load_list("{}/m_genre.txt".format(input_dir))
actor_list = load_list("{}/m_actor.txt".format(input_dir))
director_list = load_list("{}/m_director.txt".format(input_dir))
gender_list = load_list("{}/m_gender.txt".format(input_dir))
age_list = load_list("{}/m_age.txt".format(input_dir))
occupation_list = load_list("{}/m_occupation.txt".format(input_dir))
zipcode_list = load_list("{}/m_zipcode.txt".format(input_dir))
len(rate_list), len(genre_list), len(actor_list), len(director_list), len(gender_list), len(age_list), len(occupation_list), len(zipcode_list)


def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = str(rate_list.index(str(row['rate'])))
    genre_idx = []
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx.append(str(idx))

    # genres = "".join(genre_idx)

    director_idx = []
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx.append(str(idx))  # id starts from 1, not index

    # directors = "".join(director_idx)

    actor_idx = []
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx.append(str(idx))

    # actors = "".join(actor_idx)
    return rate_idx, genre_idx, director_idx, actor_idx

# -----------------------------------------------------
# item feature & relationship: movie-actor; movie-director

item_feat_dict = {}
m_directors = {}
m_actors = {}

item_id_list = []
rate_id_list = []
genre_ids_list = []
director_ids_list = []
actor_ids_list = []

for idx, row in item_data.iterrows():
    rate_idx, genre_idx, director_idx, actor_idx = item_converting(row, rate_list, genre_list, director_list,
                                                                   actor_list)
    m_directors[str(row["item"])] = director_idx
    m_actors[str(row["item"])] = actor_idx

    genres = "".join(genre_idx)
    directors = "".join(director_idx)
    actors = "".join(actor_idx)

    item_feat = "\t".join([str(row["item"]), rate_idx, genres, directors, actors])
    item_feat_dict[str(row["item"])] = item_feat

    # for generating the item_feat_df
    item_id_list.append(str(row["item"]))
    rate_id_list.append(rate_idx)
    genre_ids_list.append(genres)
    director_ids_list.append(directors)
    actor_ids_list.append(actors)

# generate dataframe for item feat
item_feat_df = pd.DataFrame({
    "item": item_id_list,
    "rating": rate_id_list,
    "genre": genre_ids_list,
    "director": director_ids_list,
    "actor": actor_ids_list
})

print("pause")


# reverse_dict
def reverse_dict(d):
    # {1:[a,b,c], 2:[a,f,g],...}
    re_d = collections.defaultdict(list)
    for k, v_list in d.items():
        for v in v_list:
            re_d[v].append(k)
    return dict(re_d)


a_movies = reverse_dict(m_actors)
d_movies = reverse_dict(m_directors)
print(len(a_movies), len(d_movies))

print("pause")

def item_suffix(x):
    return "i_"+x

def actor_suffix(x):
    return "a_"+x

def director_suffix(x):
    return "d_"+x

# ?????????rate_id_list ?????????item_id2rate_id, ??????item_id_list???index?????????????????????
iid2rate_id = dict(zip(item_id_list, rate_id_list))
iid2genre_ids = dict(zip(item_id_list, genre_ids_list))
iid2director_ids = dict(zip(item_id_list, director_ids_list))
iid2actor_ids = dict(zip(item_id_list, actor_ids_list))


# build graph
i_vertices = list(map(item_suffix, item_data.item.values.astype(str)))
a_vertices = list(map(actor_suffix, np.array(range(len(actor_list))).astype(str)))

# add edges
ia_edges = []
for iid, aids in m_actors.items():
    for aid in aids:
        ia_edges.append(("i_"+iid, "a_"+aid))

g_ia = igraph.Graph()
g_ia.add_vertices(i_vertices)
g_ia.add_vertices(a_vertices)
g_ia.add_edges(ia_edges)

print("pause")

idx2string_ia = dict(zip(range(len(g_ia.vs["name"])), g_ia.vs["name"]))
string2idx_ia = dict(zip(g_ia.vs["name"], range(len(g_ia.vs["name"]))))
MAX_LENGTH_IA = 30

def mapGid2Iid_ia(x):
    # x is a list of lists
    y = idx2string_ia[x]
    y = y.split("_")[1]
    return y

gift_ia_iids = []
gift_ia_ratings = []
gift_ia_genres = []
gift_ia_directors = []
gift_ia_actors = []
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

    # ?????????????????????, ???????????????sequence????????????
    # feature ??? rating, actors, directors,
    rating_seq = []
    genre_seq = []
    director_seq = []
    actor_seq = []

    for iid in iids:

        rating_seq.append(iid2rate_id[iid])
        genre_seq.append(iid2genre_ids[iid])
        director_seq.append(iid2director_ids[iid])
        actor_seq.append(iid2actor_ids[iid])

    iids = ",".join(iids)
    rating_seq = ",".join(rating_seq)
    genre_seq = ",".join(genre_seq)
    director_seq = ",".join(director_seq)
    actor_seq = ",".join(actor_seq)

    gift_ia_iids.append(iids)
    gift_ia_ratings.append(rating_seq)
    gift_ia_genres.append(genre_seq)
    gift_ia_directors.append(director_seq)
    gift_ia_actors.append(actor_seq)
    gift_ia_length.append(num)

# generate dataframe for item feat
gift_ia_df = pd.DataFrame({
    "item": item_id_list,
    "gift_ia_item": gift_ia_iids,
    "gift_ia_rating": gift_ia_ratings,
    "gift_ia_genre": gift_ia_genres,
    "gift_ia_director": gift_ia_directors,
    "gift_ia_actor": gift_ia_actors,
    "gift_ia_length": gift_ia_length
})



# for item director graph
d_vertices = list(map(director_suffix, np.array(range(len(director_list))).astype(str)))
id_edges = []
for iid, dids in m_directors.items():
    for did in dids:
        id_edges.append(("i_"+iid, "d_"+did))

g_id = igraph.Graph()
g_id.add_vertices(i_vertices)
g_id.add_vertices(d_vertices)
g_id.add_edges(id_edges)

idx2string_id = dict(zip(range(len(g_id.vs["name"])), g_id.vs["name"]))
string2idx_id = dict(zip(g_id.vs["name"], range(len(g_id.vs["name"]))))
MAX_LENGTH_ID = 30

def mapGid2Iid_id(x):
    # x is a list of lists
    y = idx2string_id[x]
    y = y.split("_")[1]
    return y

gift_id_iids = []
gift_id_ratings = []
gift_id_genres = []
gift_id_directors = []
gift_id_actors = []
gift_id_length = []

for iid in i_vertices:

    gid = string2idx_id[iid] # use for reduce itself
    g_aids = g_id.neighbors(vertex = iid)
    g_iids = g_id.neighborhood(vertices=g_aids)
    g_iids = np.concatenate(g_iids, axis=0)
    g_iids = list(set(g_iids).difference(set(g_aids)).difference([gid]))
    iids = list(map(mapGid2Iid_id, g_iids))
    iids = np.random.permutation(iids)

    if len(iids)> MAX_LENGTH_ID:
        iids = iids[:MAX_LENGTH_ID]
        num = MAX_LENGTH_ID
    else:
        num = len(iids)

    # ?????????????????????, ???????????????sequence????????????
    # feature ??? rating, actors, directors,
    rating_seq = []
    genre_seq = []
    director_seq = []
    actor_seq = []

    for iid in iids:
        rating_seq.append(iid2rate_id[iid])
        genre_seq.append(iid2genre_ids[iid])
        director_seq.append(iid2director_ids[iid])
        actor_seq.append(iid2actor_ids[iid])

    iids = ",".join(iids)
    rating_seq = ",".join(rating_seq)
    genre_seq = ",".join(genre_seq)
    director_seq = ",".join(director_seq)
    actor_seq = ",".join(actor_seq)

    gift_id_iids.append(iids)
    gift_id_ratings.append(rating_seq)
    gift_id_genres.append(genre_seq)
    gift_id_directors.append(director_seq)
    gift_id_actors.append(actor_seq)
    gift_id_length.append(num)

# generate dataframe for item feat
gift_id_df = pd.DataFrame({
    "item": item_id_list,
    "gift_id_item": gift_id_iids,
    "gift_id_rating": gift_id_ratings,
    "gift_id_genre": gift_id_genres,
    "gift_id_director": gift_id_directors,
    "gift_id_actor": gift_id_actors,
    "gift_id_length": gift_id_length
})

print("pause")

# store
gift_df = pd.merge(left=gift_ia_df, right=gift_id_df, on=["item"], sort=False, how="inner")
gift_df.to_csv(input_dir+"gift_df.csv", index=False)
