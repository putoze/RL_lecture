# Solution for Rabbit Farm assignment
#

infile_name='rabbit_farm.txt'

#open input/open files
infile=open(infile_name,'r')
sep=':'

#initialize variables, list, dict for accumulating results
money_total=0
money_list=[]
rabbit_dict={'rabbit':0, 'rabbot':0}
farm_items=['tiger','cabbage','carrot']

#read and process all lines from the input file
for line in infile.readlines():
    line=line.strip()  #strip leading and trailing whitespace/end-of-line's
    
    if line.isnumeric():
        #first need to check if this is an integer item (money)
        money_list.append(int(line))
        money_total+=int(line)
    else:
        #next, we process through the non-numeric strings
        # (if necessary, split strings on separator)  
        str_list=[]
        if sep in line:
            str_list=line.split(sep)
        else:
            str_list=[line]
            
        #now, look for rabbits!
        for str_item in str_list:
            if str_item not in farm_items:
                #by process of elimination, this must be a rabbit of some sort
                if 'o' in str_item:
                    #it's a rabbot
                    rabbit_dict['rabbot']+=1
                else:
                    #it's a rabbit
                    rabbit_dict['rabbit']+=1
            
#close files
infile.close() 

#write out results to file rabbits.out
# 1. sorted list of int's (money) first (largest to smallest)
# 2. total sum of integers (total money)
# 3. rabbit count dict
afile=open('rabbits.out','w')

money_list.sort(reverse=True)
afile.write(str(money_list)+'\n')

afile.write(str(money_total)+'\n')

afile.write(str(rabbit_dict)+'\n')

afile.close()