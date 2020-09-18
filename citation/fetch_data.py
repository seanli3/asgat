from citation import get_dataset


for data in ['Cora', 'CiteSeer', 'PubMed', 'nell.0.1', 'nell.0.01', 'nell.0.001', 'CS', 'Physics', 'Flickr', 'Yelp']:
    print('fetching '+ data)
    get_dataset(data)

