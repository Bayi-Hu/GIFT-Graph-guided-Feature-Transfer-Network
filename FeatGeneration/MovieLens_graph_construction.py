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


g = igraph.Graph()
m_vertex =