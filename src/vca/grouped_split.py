"""Deterministic group split: identical normalized prompts or image IDs cannot cross folds."""
import hashlib,json,re
from collections import defaultdict

def normalized_prompt(record):return ' '.join(re.findall(r'\w+',str(record.get('prompt','')).casefold()))

def image_keys(entry):
    record=entry['record'];paths=[record.get('image_path'),record.get('image'),record.get('ref_image_path')]
    # Archives distinguish common filenames such as 0001.webp. Two dataset
    # mirrors with identical archive names intentionally share the namespace.
    archives={str(s.get('archive','')).casefold() for s in entry.get('sources',[]) if s.get('archive')}
    return [a+'|'+path.replace('\\','/').casefold() for a in sorted(archives) for path in paths if isinstance(path,str) and path]

def make_groups(entries):
    parent=list(range(len(entries)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[b]=a
    by_prompt={};by_image={}
    for i,e in enumerate(entries):
        prompt=normalized_prompt(e['record'])
        if prompt:
            if prompt in by_prompt:union(i,by_prompt[prompt])
            else:by_prompt[prompt]=i
        for image in image_keys(e):
            if image in by_image:union(i,by_image[image])
            else:by_image[image]=i
    groups=defaultdict(list)
    for i,e in enumerate(entries):groups[find(i)].append(e['record_sha256'])
    return [sorted(v) for v in groups.values()]

def grouped_split(entries,validation_fraction=.05):
    if not 0<validation_fraction<1:raise ValueError('Invalid split fraction')
    groups=make_groups(entries)
    ordered=sorted(groups,key=lambda g:hashlib.sha256(''.join(g).encode()).hexdigest())
    target=max(1,round(len(entries)*validation_fraction));selected=[];count=0
    for g in ordered:
        if count>=target:break
        selected.extend(g);count+=len(g)
    validation=set(selected);all_ids={e['record_sha256'] for e in entries}
    train=all_ids-validation
    return {'train':sorted(train),'validation':sorted(validation),'groups':len(groups),
            'grouping':'transitive equal normalized prompt OR same archive-scoped image path',
            'assignment':'SHA-256 group ordering; 5% target, approximate because groups stay intact',
            'classification':'implementation-assumption; not the unpublished paper split',
            'limitation':'image paths rather than decoded-image hashes; near-duplicate visual content not excluded'}

def verify_no_leak(entries,split):
    a,b=set(split['train']),set(split['validation'])
    if a&b:raise ValueError('Record IDs overlap')
    pa={normalized_prompt(e['record']) for e in entries if e['record_sha256'] in a}
    pb={normalized_prompt(e['record']) for e in entries if e['record_sha256'] in b}
    ia={k for e in entries if e['record_sha256'] in a for k in image_keys(e)}
    ib={k for e in entries if e['record_sha256'] in b for k in image_keys(e)}
    if (pa-{''}) & (pb-{''}) or ia&ib:raise ValueError('Prompt or image identifiers overlap')
    return {'record_overlap':0,'normalized_prompt_overlap':0,'archive_image_id_overlap':0}
