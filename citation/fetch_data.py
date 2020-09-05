from citation import get_dataset


for data in ['Cora', 'Citeseer', 'PubMed', 'Reddit', 'nell.0.1', 'nell.0.01', 'nell.0.001']:
    get_dataset(data)

