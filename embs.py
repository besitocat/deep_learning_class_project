from tqdm import tqdm
import codecs
import numpy as np


def loadFastTextEmb():
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open('wiki.simple.vec', encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        fword = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[fword] = coefs
    print('found %s word vectors' % len(embeddings_index))
    return embeddings_index
   
    
if __name__ == "__main__":
    embeddingDic= loadFastTextEmb()
    embedding = embeddingDic['ooov']
    print embedding