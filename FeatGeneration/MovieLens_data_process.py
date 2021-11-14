#-*- coding:utf-8 -*-
import pandas as pd
import collections
import torch
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
print(len(ui_data))

# read item/user feature
user_data = pd.read_csv(input_dir+'users.txt', names=['user', 'gender', 'age', 'occupation_code', 'zip'],
                        sep="::", engine='python')

item_data = pd.read_csv(input_dir+'movies_extrainfos.txt', names=['item', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'],
                        sep="::", engine='python', encoding="utf-8")

user_list = list(set(ui_data.user.tolist()) | set(user_data.user))
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

# TODO: user sequence, user feature (user id, gender, occupation_code, zip), item feature
#

def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    user_idx = str(row["user"])
    gender_idx = str(gender_list.index(str(row['gender'])))
    age_idx = str(age_list.index(str(row['age'])))
    occupation_idx = str(occupation_list.index(str(row['occupation_code'])))
    zip_idx = str(zipcode_list.index(str(row['zip'])[:5]))

    return user_idx, gender_idx, age_idx, occupation_idx, zip_idx
    # return "\t".join([user_idx, gender_idx, age_idx, occupation_idx, zip_idx])


# user feature -----------------------------------------------------
user_feat_dict = {}
user_id_list = []
gender_id_list = []
age_id_list = []
occupation_id_list = []
zip_id_list = []

for idx, row in user_data.iterrows():

    user_idx, gender_idx, age_idx, occupation_idx, zip_idx = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
    user_feat = "\t".join([user_idx, gender_idx, age_idx, occupation_idx, zip_idx])

    user_feat_dict[str(row["user"])] = user_feat

    # for generating the user_feat_df
    user_id_list.append(user_idx)
    gender_id_list.append(gender_idx)
    age_id_list.append(age_idx)
    occupation_id_list.append(occupation_idx)
    zip_id_list.append(zip_idx)

# generate dataframe for user feat
user_feat_df = pd.DataFrame({
    "user": user_id_list,
    "gender": gender_id_list,
    "age": age_id_list,
    "occupation": occupation_id_list,
    "zip": zip_id_list
})

print("pause")

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

# TODO: 处理 ui_data, user_sequence
# postive sample
pos_ui_data = ui_data[ui_data.rating>=4].copy()
# confine the timestamp of item in sequence less than that of target item
X = pd.merge(left=ui_data, right=pos_ui_data, how='left', on=["user"], sort=False)
X = X[X.timestamp_x > X.timestamp_y]

# group by according to user and timestamp_x and sort the sequence based on timestamp_y

def udf(df):

    def takeSecond(elem):
        return elem[1]

    # output = []
    iid_sequence = []
    X = list(zip(df.item_y, df.timestamp_y))
    X.sort(key=takeSecond, reverse=True)
    length = 0
    for x in X: # set max length to 100
        iid_sequence.append(str(x[0]))
        length += 1
        if length>= 100:
            break

    data = pd.DataFrame({
        "user": [str(df.user.iloc[0])],
        "item": [str(df.item_x.iloc[0])],
        "rating": [str(df.rating_x.iloc[0])],
        "time": [str(df.timestamp_x.iloc[0])],
        "length": [str(length)],
        "sequence": [",".join(iid_sequence)]
    })

    return data

ui_data_sample = X.groupby(["user", "item_x", "timestamp_x", "rating_x"]).apply(udf).values
ui_data_new = pd.DataFrame(ui_data_sample.values, columns=["user", "item", "rating", "time", "length", "sequence"])

# pd.merge(left=ui_data_new, right= , )


