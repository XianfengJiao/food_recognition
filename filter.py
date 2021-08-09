import pickle as pkl
import os

root = "./mini_food_data_matched"
name = "valid"
lis = pkl.load(open(f'{root}/data_90/{name}_lis.pkl', 'rb'))
lis_toy = open(f'{root}/data_90/{name}_lis_toy.pkl', 'wb')

toy = []

for i in lis:
    try:
        open(f"{root}/FoodImages/{i}.jpg", "r")
        open(f"{root}/recipe_feat_new/{i}/{i}_title.npy")
    except Exception:
        continue
    else:
        toy.append(i)
"""
for i in lis:
    tmp_0 = []
    for j in i:
        tmp_1 = []
        for k in j:
            try:
                open(f"{root}/FoodImages/{k}.jpg", "r")
                open(f"{root}/recipe_feat_new/{k}/{k}_title.npy")
            except Exception:
                continue
            else:
                tmp_1.append(k)
        tmp_0.append(tmp_1)
    toy.append(tmp_0)
"""
pkl.dump(toy, lis_toy)
lis_toy.close()
