import json
import os
with open("./classids_miniKinetics200_s128.json") as file:
    json_data = json.load(file)
    json_sort = sorted(json_data.items(), key=lambda item:item[1])
    list200 = os.listdir("/ws/data/")
    for index, value in enumerate(list200):
        if "_" in value:
            a = "\"" + value + "\""
            list200[index] = a
    l = 0
    new_json = dict()
    for i in json_sort:
        if i[0] in list200:
            k = str(i[0])
            new_json[k] = l
            l += 1

with open("classids_miniKinetics200_s128_real.json", 'w') as json_file:
    json.dump(new_json, json_file)