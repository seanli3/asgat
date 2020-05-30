import numpy as np

parameters = []
for dropout in np.arange(0.1, 1, 0.1): # 0.9 done
    for hidden in range(64, 65, 16): # 134 doing
        for heads in range(8, 9, 2): # 18 done
            for data in ['Citeseer']:
                parameters.append('--dataset={} --alpha=0.2 --lr=0.001 --weight_decay=0.0005 --hidden={} --heads={} --dropout={} --cuda --chebyshev_order=16 --early_stopping=10\n'\
                                          .format(data, str(hidden), str(heads), str(dropout)))

with open('parameters.txt', 'w') as the_file:
    the_file.writelines(parameters)

