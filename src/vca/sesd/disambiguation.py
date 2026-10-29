import re

def explicit_sense(term,feedback,entries):
    hits=[]
    for e in entries:
        if e['category']!='Polysemous Words' or e['term'].lower()!=term.lower():continue
        matched=[a for a in [e['sense'],*e['aliases']] if re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',feedback,re.I)]
        if matched:hits.append((max(len(a) for a in matched),e))
    if not hits:return None
    hits.sort(key=lambda x:-x[0])
    # Conflicting equally strong explicit senses require a question.
    if len(hits)>1 and hits[0][0]==hits[1][0]:return None
    return hits[0][1]

def ambiguities(prompt,entries,resolved=None):
    resolved=resolved or {};result={}
    for e in entries:
        term=e['term'].lower()
        if e['category']=='Polysemous Words' and re.search(r'\b'+re.escape(term)+r'\b',prompt,re.I):
            if term not in resolved and not explicit_sense(term,prompt,entries):
                result.setdefault(term,[]).append(e['sense'])
    return result
