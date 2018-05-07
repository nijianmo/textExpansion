import pickle
import os

path = "tst.txt"
with open(path, 'r') as f:
  tsts = f.readlines()
tsts = [tst.strip() for tst in tsts]

# the number of distinct unigrams and bigrams (respectively) in generated responses, divided by the total number of generated tokens
total = 0
uni = set()
bi = set()
for tst in tsts:
  l = tst.split()
  total += len(l)
  for w in l:
    uni.add(w)
  for (w1,w2) in zip(l[:-1],l[1:]):
    bi.add((w1,w2))
   
print(len(uni))
print(len(bi))
print(total) 
div1 = len(uni) / total 
div2 = len(bi) / total 

print("div1/div2 = {}/{}".format(div1, div2))  

