from webkb import get_dataset


for data in ["Cornell", 'Texas', "Washington", "Wisconsin"]:
    print('fetching '+ data)
    get_dataset(data)

