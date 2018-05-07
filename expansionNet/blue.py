# BLEU
from nltk.translate.bleu_score import sentence_bleu

reference = "as a professional photographer , my lenses are my life - thought i 'd try a less expensive but still high quality lens from rokinon and so far seems to be working well for me .".split()
candidates = ["lumix 70d dslr could not take incredibly sharp photo frames - with <unk> shots - <unk> full sharpness - not web , fall fixed , window - but other than that great close-up experience .".split(), "results now i 've been very happy with this lens . it is a little bigger than i thought , but i have n't had any issues with the lens so far . i just wish it was a little more sturdy and i 'd recommend it to anyone .".split()]

for candidate in candidates:
  s1 = sentence_bleu([reference], candidate, weights=(1.0,))
  s4 = sentence_bleu([reference], candidate)
  print("bleu-1={} / blue-4= {}".format(s1, s4))
