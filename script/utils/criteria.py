import nltk

# calculate the ngram of BLEU score
####### input ########
# reference: reference sentence lists Array (eg:[[1,2,3],[4,5,6]])
# predict:   predict sentence list
# n:         ngram
##### output ########
# BLEUscore: BLEU-n score

def bleu_score(reference,predict,n = 1):
    BLEUscore = nltk.translate.bleu_score.modified_precision(reference,predict,n)
    return float(BLEUscore)
