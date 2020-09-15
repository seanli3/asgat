from entities import get_dataset


for data in ["WikiCS"]:
    print('fetching '+ data)
    get_dataset(data)

