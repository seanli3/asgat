from citation import get_planetoid_dataset


for data in ['Cora', 'Citeseer', 'Pubmed']:
    get_planetoid_dataset(data)

