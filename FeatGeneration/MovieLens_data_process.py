import pandas as pd
import torch
import re


def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    genre_idx = torch.zeros(1, 25).long()
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1  # one-hot vector

    director_idx = torch.zeros(1, 2186).long()
    director_id = []
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
        director_id.append(idx + 1)  # id starts from 1, not index
    actor_idx = torch.zeros(1, 8030).long()
    actor_id = []
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
        actor_id.append(idx + 1)
    return torch.cat((rate_idx, genre_idx), 1), torch.cat((rate_idx, genre_idx, director_idx, actor_idx),
                                                          1), director_id, actor_id


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)  # (1, 4)


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

