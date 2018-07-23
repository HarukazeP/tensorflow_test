# -*- coding: utf-8 -*-
'''
空所内に何単語あるか調べるやつ
'''


len1=0
len2=0
len3=0
len4=0
len5=0
len6=0





with open('../../Data/my_nmt/center_choices.txt') as f:
    for line in f:
        line=line[line.find('{')+2:line.rfind('}')-2]
        tmp=line.split(' ### ')
        for x in tmp:
            if x.find(' ')>=0:
                xlen=len(x.split(' '))
            else:
                xlen=1

            if xlen==1:
                len1+=1
            elif xlen==2:
                len2+=1
            elif xlen==3:
                len3+=1
            elif xlen==4:
                len4+=1
            elif xlen==5:
                len5+=1
            elif xlen==6:
                len6+=1
            else:
                print(x)

sum=len1+len2+len3+len4+len5
print(len1)
print(len2)
print(len3)
print(len4)
print(len5)
print(len6)

print(1.0*len1/sum)
print(1.0*len2/sum)
print(1.0*len3/sum)
print(1.0*len4/sum)
print(1.0*len5/sum)


