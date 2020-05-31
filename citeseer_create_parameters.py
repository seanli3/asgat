import numpy as np

parameters = []
# dropout=0.95 hidden=24 heads=8 alpha=0.2 lr=0.0001 weight_decay=
for wd in [0.0001, 0.00005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]: # 8 done
    parameters.append('--dataset=Citeseer --alpha=0.2 --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --cuda --chebyshev_order=16 --early_stopping=10 --epochs=10000 --weight_decay={} --chebyshev_order=16\n'.format(str(wd)))

with open('citeseer_parameters.txt', 'w') as the_file:
    the_file.writelines(parameters)

