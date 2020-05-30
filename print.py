import os
import glob
import operator

files = glob.glob('./*.out')
acc = {}
for f in files:
    with open(f, "r") as file:
        first_line = file.readline()
        for last_line in file:
            pass
        try:
            acc[file.name] = float(last_line.split(' ')[-1])
        except:
            pass

max_acc = max(acc.values())
print(max_acc)

for fname in acc:
    if acc[fname] >= max_acc:
        print(fname)
        with open(fname, "r") as file:
            j = 0
            for last_line in file:
              if j <= 37 and j > 32:
                print(last_line)
              j+=1
            print(last_line)


# file_name = max(acc.items(), key=operator.itemgetter(1))[0]
# print(file_name)
# 
# with open(file_name, "r") as file:
#     j = 0
#     for last_line in file:
#       if j <= 37 and j > 30:
#         print(last_line)
#       j+=1
#     print(last_line)


