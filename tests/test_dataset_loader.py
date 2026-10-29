import hashlib,json,pytest
from vca.dataset import load_catalog,episode_from_record
from vca.data_audit import record_entry

def test_verified_catalog_loader(tmp_path):
    record={'prompt':'a cat','question1':'q','answer1':'red','question2':'q','answer2':'on a desk',
            'question3':'q','answer3':'a hat'}
    e=record_entry(record,'x.json','x');e.update(record_id=e['record_sha256'][:16],sources=[{'source':'test_fixture'}],split='train')
    p=tmp_path/'catalog.jsonl';p.write_text(json.dumps(e)+'\n',encoding='utf-8')
    loaded=load_catalog(p,'train',3);episode=episode_from_record(loaded[0])
    assert episode['prompt']=='a cat' and episode['feedbacks']==['red','on a desk','a hat']
    assert not load_catalog(p,'validation')
    e['record']['prompt']='tampered';p.write_text(json.dumps(e),encoding='utf-8')
    with pytest.raises(ValueError,match='hash'):load_catalog(p)
