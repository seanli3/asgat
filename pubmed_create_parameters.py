import numpy as np

parameters = []
# dropout=0.95 hidden=24 heads=8 alpha=0.2 lr=0.0001 weight_decay=
for hidden in range(8, 65, 8): # 8 done
    parameters.append('--dataset=Pubmed --alpha=0.2 --lr=0.0001 --hidden={} --heads=8 --dropout=0.95 --cuda --chebyshev_order=16 --early_stopping=10 --epochs=10000 --weight_decay=0.0005 --chebyshev_order=16\n'.format(str(hidden)))

with open('pubmed_parameters.txt', 'w') as the_file:
    the_file.writelines(parameters)

