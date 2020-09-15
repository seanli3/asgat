from webkb import get_dataset


for data in ['AIFB', 'MUTAG', 'BGS', 'AM']:
    print('fetching '+ data);
    get_dataset(data)

