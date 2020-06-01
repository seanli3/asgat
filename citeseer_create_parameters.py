import numpy as np

parameters = []
# dropout=0.95 hidden=24 heads=8 alpha=0.2 lr=0.0001 weight_decay=0.001 cheby_order=12
for order in [12]: # 12 done
    parameters.append('--dataset=Citeseer --alpha=0.2 --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --cuda --chebyshev_order=12 --early_stopping=10 --epochs=8000 --weight_decay=0.001\n'.format(str(order)))

with open('citeseer_parameters.txt', 'w') as the_file:
    the_file.writelines(parameters)

