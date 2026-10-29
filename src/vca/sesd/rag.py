"""Actual local TF-IDF vectors and cosine retrieval, with term and source evidence."""
import re
from collections import Counter
import numpy as np

def terms(text):return re.findall(r'\w+',text.lower())

class VectorRetriever:
    def __init__(self,kb,embedder=None):self.kb=kb;self.embedder=embedder;self.rebuild()

    def rebuild(self):
        self.docs=[terms(' '.join([e['term'],e['sense'],e['text'],*e['aliases']])) for e in self.kb.entries]
        self.vocab={w:i for i,w in enumerate(sorted(set(sum(self.docs,[]))))}
        df=Counter(w for doc in self.docs for w in set(doc))
        self.idf=np.array([np.log((1+len(self.docs))/(1+df[w]))+1 for w in self.vocab])
        self.matrix=np.stack([self.vectorize(' '.join(d)) for d in self.docs])
        if self.embedder is not None:
            vectors=self.embedder([' '.join(d) for d in self.docs]).detach().cpu().float().numpy()
            self.semantic_matrix=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
        self.version=self.kb.version

    def vectorize(self,text):
        vec=np.zeros(len(self.vocab));counts=Counter(terms(text))
        for w,n in counts.items():
            if w in self.vocab:vec[self.vocab[w]]=(1+np.log(n))*self.idf[self.vocab[w]]
        norm=np.linalg.norm(vec)
        return vec/norm if norm else vec

    def search(self,query,k=8,category=None):
        if self.version!=self.kb.version:self.rebuild()
        if self.embedder is None:
            q=self.vectorize(query);scores=self.matrix@q
        else:
            q=self.embedder([query]).detach().cpu().float().numpy()[0];q=q/np.linalg.norm(q)
            scores=self.semantic_matrix@q
        query_terms=set(terms(query));rows=[]
        for idx in np.argsort(-scores,kind='stable'):
            e=self.kb.entries[int(idx)]
            if scores[idx]<=0 or (category and e['category']!=category):continue
            rows.append({**e,'score':float(scores[idx]),'matched_terms':sorted(query_terms & set(self.docs[idx])),
                'retrieval_method':('CLIP text embedding cosine (implementation-assumption)' if self.embedder else 'local TF-IDF cosine (implementation-assumption)'),
                'query':query,'query_embedding_truncation':'CLIP max 77 tokens; latest feedback placed first' if self.embedder else None})
            if len(rows)>=k:break
        return rows
