from citation import get_planetoid_dataset


for data in ['Cora', 'Citeseer', 'PubMed']:
    get_planetoid_dataset(data)

