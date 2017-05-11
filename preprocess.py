import re
import os
import operator

from random import shuffle

def get_name_data(data_dir):
    first_names = {}
    fn2country = {}
    last_names = {}
    ln2country = {}
    new_names = {}

    for root, dir, files in os.walk(data_dir):
        for file_cnt, file_name in enumerate(sorted(files)):
            data = open(os.path.join(root, file_name))
            file_len = 0

            if file_name == 'name_to_country.txt':
                for k, line in enumerate(data):
                    raw_name, nationality = line[:-1].split('\t')
                    raw_name = re.sub(r'\ufeff', '', raw_name)    # delete BOM
                    split_name = raw_name.split()
                    if len(split_name) == 2:
                        first_name, last_name = split_name
                        if first_name not in first_names:
                            first_names[first_name] = 0
                            fn2country[first_name] = nationality
                        if last_name not in last_names:
                            last_names[last_name] = 0
                            ln2country[last_name] = nationality
                        first_names[first_name] += 1
                        last_names[last_name] += 1
                    
                    file_len = k + 1
            else:
                pass 

            if file_len > 0:
                print('reading', file_name, 'of length', file_len)
    
    print('fn', len(first_names), 'ln', len(last_names))
    fn_sorted = sorted(first_names.items(), key=operator.itemgetter(1))[::-1][:10]
    print('first names', fn_sorted)
    ln_sorted = sorted(last_names.items(), key=operator.itemgetter(1))[::-1][:10]
    print('last names', ln_sorted)

    for fn in first_names:
        for ln in last_names:
            if fn2country[fn] == ln2country[ln]:
                new_names[fn + ' ' + ln] = fn2country[fn]

    print('nn', len(new_names))
    print('new names', list(new_names.items())[:10])

    return new_names


data_to_write = get_name_data('data')

f = open('./data/new_names (tmp).txt', 'w')
for name, nationality in data_to_write.items():
    f.write(name + '\t' + nationality + '\n')
f.close()

