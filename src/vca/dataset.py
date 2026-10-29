"""Consume only hash-verified records from audit_dataset.py; no guessed dialogue rows."""
from pathlib import Path
import hashlib,json

def load_catalog(path,split=None,min_turns=2):
    rows=[];seen=set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        entry=json.loads(line);record=entry['record']
        digest=hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
        if digest!=entry['record_sha256']:raise ValueError('Catalog record hash mismatch')
        if not entry.get('sources'):raise ValueError('Record provenance missing')
        if digest in seen:raise ValueError('Duplicate catalog record')
        seen.add(digest)
        if split and entry.get('split')!=split:continue
        if len(entry.get('qa_pairs',[]))<min_turns:continue
        if not isinstance(record.get('prompt'),str) or not record['prompt'].strip():continue
        rows.append(entry)
    return rows

def episode_from_record(entry,rounds=3):
    pairs=entry['qa_pairs'][:rounds]
    if len(pairs)<rounds:raise ValueError('Insufficient non-placeholder Q/A turns')
    return {'prompt':entry['record']['prompt'],'feedbacks':[p['answer'] for p in pairs],
        'questions':[p.get('question') for p in pairs],'record_id':entry['record_id'],
        'record_sha256':entry['record_sha256'],'sources':entry['sources'],
        'expected_annotation_prompts':[p.get('refined_prompt') for p in pairs],
        'annotation_use':'original prompt and ordered actual answers; refined prompt fields are evidence, not forced LLM output'}
