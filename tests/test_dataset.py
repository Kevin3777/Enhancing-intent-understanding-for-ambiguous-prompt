import importlib.util,io,json,zipfile
from pathlib import Path
import sys
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
from audit_dataset import audit_zip

def test_dataset_audit_no_pickle_execution():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('a.json',json.dumps({'dialogue_id':'x','initial_prompt':'a cat','turns':[{'text':'a'},{'text':'b'}]}))
        z.writestr('a.png',b'fixture, not an actual PNG');z.writestr('danger.pkl',b'do not execute')
    with zipfile.ZipFile(buf) as z:r=audit_zip(z)
    assert r['json_record_count']==1 and r['image_count']==1 and r['dialogue_like_record_count']==1
    assert r['pickle_files_not_opened']==['danger.pkl'] and r['observed_json_fields']==['dialogue_id','initial_prompt','turns']

def test_mapping_and_qa_columns():
    from vca.data_audit import normalized_records,qa_pairs
    records=normalized_records({'001.webp':{'prompt':'a cat','question1':'what','answer1':'cat','question2':'where','answer2':'Enter the answer 2 here'}})
    assert records[0][0]=='001.webp'
    pairs,placeholders=qa_pairs(records[0][1]);assert len(pairs)==1 and placeholders==1

def test_nested_archives_are_audited_without_extracting():
    child=io.BytesIO()
    with zipfile.ZipFile(child,'w') as z:
        z.writestr('r.json',json.dumps([{'prompt':'a cat','question1':'x','answer1':'cat','question2':'y','answer2':'red'}]))
        z.writestr('tool.exe',b'never execute')
    outer=io.BytesIO()
    with zipfile.ZipFile(outer,'w') as z:z.writestr('nested.zip',child.getvalue())
    with zipfile.ZipFile(outer) as z:r=audit_zip(z)
    assert r['unique_annotation_records']==1 and r['unique_dialogue_like_records']==1
    assert r['nested_archives'][0]['executables_not_run']==['tool.exe']
