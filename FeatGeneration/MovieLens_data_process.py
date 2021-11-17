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

input_dir = 'MovieLens-1M/'

# read logs
ui_data = pd.read_csv(input_dir+'ratings.txt', names=['user', 'item', 'rating', 'timestamp'],sep="::", engine='python')

ui_data.loc[ui_data[ui_data.rating < 4].index, 'label'] = "0"
ui_data.loc[ui_data[ui_data.rating >= 4].index, 'label'] = "1"

ui_data = ui_data.astype("str")

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
user_id_list = []
gender_id_list = []
age_id_list = []
occupation_id_list = []
zip_id_list = []

for idx, row in user_data.iterrows():

    user_idx, gender_idx, age_idx, occupation_idx, zip_idx = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)

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
pos_ui_data = ui_data[ui_data.label=="1"].copy()

# pos_ui_item_feat = pd.merge(left=pos_ui_data[["user", "item", "timestamp"]],
#                             right=item_feat_df, on=["item"], how="left", sort=False)
#
#
# # confine the timestamp of item in sequence less than that of target item
# X = pd.merge(left=ui_data[["user", "item", "timestamp", "label"]], right=pos_ui_item_feat, how='left', on=["user"], sort=False)
# X = X[X.timestamp_x > X.timestamp_y]
#
# # group by according to user and timestamp_x and sort the sequence based on timestamp_y
#
# def udf(df):
#     def takeFirst(elem):
#         return elem[0]
#
#     # output = []
#     item_seq = []
#     rating_seq = []
#     genre_seq = []
#     director_seq = []
#     actor_seq = []
#
#     X = list(zip(df.timestamp_y, df.item_y, df.rating, df.genre, df.director, df.actor))
#     X.sort(key=takeFirst, reverse=True)
#     length = 0
#     for x in X:  # set max length to 100
#         item_seq.append(str(x[1]))
#         rating_seq.append(str(x[2]))
#         genre_seq.append(str(x[3]))
#         director_seq.append(str(x[4]))
#         actor_seq.append(str(x[5]))
#
#         length += 1
#         if length >= 50:
#             break
#
#     return np.array([[df.iloc[0]["user"], df.iloc[0]["item_x"], df.iloc[0]["timestamp_x"], df.iloc[0]["label"],
#                       str(length), ",".join(item_seq), ",".join(rating_seq), ",".join(genre_seq),
#                       ",".join(director_seq), ",".join(actor_seq)]])
#
#
# X_ = X.groupby(["user", "item_x", "timestamp_x", "label"]).apply(udf)
# ui_data_new = pd.DataFrame(np.concatenate(X_.values, axis=0), columns=["user", "item", "timestamp", "label", "length", "item_seq", "rating_seq", "genre_seq", "director_seq", "actor_seq"])


# 第二种策略是不构建复杂的sequence特征
pos_ui_item_feat = pd.merge(left=pos_ui_data[["user", "item", "timestamp"]],
                            right=item_feat_df[["item", "rating", "genre"]], on=["item"], how="left", sort=False)


# confine the timestamp of item in sequence less than that of target item
X = pd.merge(left=ui_data[["user", "item", "timestamp", "label"]], right=pos_ui_item_feat, how='left', on=["user"], sort=False)
X = X[X.timestamp_x > X.timestamp_y]

# group by according to user and timestamp_x and sort the sequence based on timestamp_y

def udf(df):
    def takeFirst(elem):
        return elem[0]

    # output = []
    item_seq = []
    rating_seq = []
    genre_seq = []
    # director_seq = []
    # actor_seq = []

    X = list(zip(df.timestamp_y, df.item_y, df.rating, df.genre))
                 # , df.director, df.actor))

    X.sort(key=takeFirst, reverse=True)
    length = 0
    for x in X:  # set max length to 100
        item_seq.append(str(x[1]))
        rating_seq.append(str(x[2]))
        genre_seq.append(str(x[3]))
        # director_seq.append(str(x[4]))
        # actor_seq.append(str(x[5]))

        length += 1
        if length >= 50:
            break

    return np.array([[df.iloc[0]["user"], df.iloc[0]["item_x"], df.iloc[0]["timestamp_x"], df.iloc[0]["label"],
                      str(length), ",".join(item_seq), ",".join(rating_seq), ",".join(genre_seq)]])
                      #, ",".join(director_seq), ",".join(actor_seq)]])


X_ = X.groupby(["user", "item_x", "timestamp_x", "label"]).apply(udf)
ui_data_new = pd.DataFrame(np.concatenate(X_.values, axis=0), columns=["user", "item", "timestamp", "label", "length", "item_seq", "rating_seq", "genre_seq"])
                                                                                                                                                 # "director_seq", "actor_seq"])



# 切分 新/老item

# pd.merge(left=ui_data_new, right= , )

# item feature used for GIFT
# target item
# gift_sequence


# 切分

