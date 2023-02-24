# import library
import re

# open txt
f = open('rabbit_farm.txt')
outfile = open('rabbits.out', 'w')

# create list and dic
sum_num = 0
farm_list = []
farm_list = f.readlines()
animal_dict = {'rabbit': 0, 'rabbot': 0}

# int list
int_list = []
for object in farm_list:
    if re.findall(r"\d+", object) != []:
        farm_list.remove(object)
        int_num = int(object)
        int_list.append(int_num)
        sum_num += int_num
int_list = sorted(int_list, reverse=True)
# print(sum_num)
# print(int_list)

# find animal
for colon in farm_list:
    animal_list = colon.split(":")
    for animal in animal_list:
        animal = animal.replace('\n', '')
        if animal in animal_dict:
            animal_dict[animal] += 1
        else:
            sorted_word = ''.join(sorted(animal))
            if sorted_word == 'abbirt':
                animal_dict['rabbit'] += 1
            elif sorted_word == 'abbort':
                animal_dict['rabbot'] += 1
# print(animal_dict)

# outfile
outfile.write(str(int_list) + '\n')
outfile.write(str(sum_num) + '\n')
outfile.write(str(animal_dict))
outfile.close
