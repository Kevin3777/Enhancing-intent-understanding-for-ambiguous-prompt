"""Independent source audit. Never unpickle data or equate images with dialogues."""
from pathlib import Path
import argparse,datetime,hashlib,io,json,re,sys,time,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fetch_assets import download
from vca.data_audit import audit_zip

def read_json(url):
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'VCA-reproduction-audit/1.0'})
            with urllib.request.urlopen(req,timeout=40) as response:return json.load(response)
        except (OSError,ValueError):
            if attempt==3:raise
            time.sleep(1+attempt)

class RemoteRangeFile(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0;self.requests=0
        req=urllib.request.Request(url,headers={'Range':'bytes=0-0'})
        with urllib.request.urlopen(req,timeout=45) as response:
            cr=response.headers.get('Content-Range','')
            if response.status!=206 or not cr:raise RuntimeError('Range requests unavailable; full archive not silently downloaded')
            self.size=int(cr.split('/')[-1]);response.read(1)
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if pos<0:raise OSError('Negative seek')
        self.pos=pos;return pos
    def read(self,size=-1):
        if size<0:size=self.size-self.pos
        size=min(size,self.size-self.pos)
        if size<=0:return b''
        if size>16*1024*1024:raise RuntimeError('Oversized remote read refused')
        start,end=self.pos,self.pos+size-1
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req,timeout=45) as response:
            cr=response.headers.get('Content-Range','')
            if response.status!=206 or not cr.startswith(f'bytes {start}-{end}/'):
                raise RuntimeError(f'Invalid range response {cr!r}')
            data=response.read(size+1)
        if len(data)!=size:raise RuntimeError('Short/oversized range response')
        self.pos+=size;self.requests+=1;return data

def main():
    p=argparse.ArgumentParser();p.add_argument('--local',type=Path);p.add_argument('--out',type=Path,default=ROOT/'outputs/dataset_audit')
    p.add_argument('--download-zenodo',action='store_true');p.add_argument('--remote-manifests',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.local:
        with zipfile.ZipFile(a.local) as z:report=audit_zip(z)
        report['sha256']=hashlib.file_digest(a.local.open('rb'),'sha256').hexdigest()
        (a.out/'local.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return
    report={'accessed_date':'2026-10-27','sources':[],'claim':'Sources audited independently; no assumption of equivalent 1673 dialogue JSONs.'}
    for repo in ('naseele/Image_prompt','Kevin3777/Enhancing_Intent_Understanding'):
        row={'source':repo,'url':'https://huggingface.co/datasets/'+repo}
        try:
            meta=read_json('https://huggingface.co/api/datasets/'+repo+'?blobs=true')
            row.update(revision=meta['sha'],gated=meta.get('gated'),license=(meta.get('cardData') or {}).get('license','not in card metadata'))
            row['files']=meta['siblings']
            (a.out/(repo.split('/')[0]+'_metadata.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
            row['archives']=[]
            for file in row['files']:
                if not file['rfilename'].endswith('.zip'):continue
                ar={'file':file['rfilename'],'lfs':file.get('lfs')}
                if a.remote_manifests:
                    try:
                        url=f"https://huggingface.co/datasets/{repo}/resolve/{meta['sha']}/{file['rfilename']}"
                        for attempt in range(4):
                            try:
                                with zipfile.ZipFile(RemoteRangeFile(url)) as z:ar.update(audit_zip(z))
                                break
                            except (OSError,ValueError,zipfile.BadZipFile):
                                if attempt==3:raise
                                time.sleep(1+attempt)
                        ar['status']='central_directory_audited'
                    except Exception as e:ar.update(status='BLOCKED',error=type(e).__name__+': '+str(e))
                row['archives'].append(ar)
            row['status']='metadata_verified'
        except Exception as e:row.update(status='BLOCKED',error=str(e))
        report['sources'].append(row)
        (a.out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(repo+' audited',flush=True)
    row={'source':'Zenodo 14876029','url':'https://zenodo.org/records/14876029'}
    try:
        meta=read_json('https://zenodo.org/api/records/14876029');row.update(metadata=meta['metadata'],files=meta['files'])
        (a.out/'zenodo_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        row['latest_version_endpoint']=meta['links']['latest']
        row['latest_version']=read_json(meta['links']['latest']).get('id')
        if a.download_zenodo:
            file=meta['files'][0];path=ROOT/'data/downloads'/file['key']
            info=download(file['links']['self'],path)
            md5=hashlib.file_digest(path.open('rb'),'md5').hexdigest()
            if 'md5:'+md5!=file['checksum']:raise ValueError('Zenodo MD5 mismatch')
            row['download']={**info,'md5_verified':md5}
            with zipfile.ZipFile(path) as z:row['archive']=audit_zip(z)
            row['status']='download_hash_and_members_verified'
        else:row['status']='metadata_verified'
    except Exception as e:row.update(status='BLOCKED',error=type(e).__name__+': '+str(e))
    report['sources'].append(row)
    # Only compare actual published content identifiers, not names or README claims.
    left,right=report['sources'][:2]
    left_hash={x['file']:(x.get('lfs') or {}).get('sha256') for x in left.get('archives',[])}
    right_hash={x['file']:(x.get('lfs') or {}).get('sha256') for x in right.get('archives',[])}
    report['hf_archive_content_matches']={k:('unverified' if not v or not right_hash.get(k) else 'matched' if right_hash[k]==v else 'different') for k,v in left_hash.items()}
    catalog={}
    for source in report['sources']:
        archives=source.get('archives',[])+([source['archive']] if 'archive' in source else [])
        for ar in archives:
            for entry in ar.get('record_entries',[]):
                digest=entry['record_sha256']
                provenance={'source':source['source'],'revision':source.get('revision'),
                    'archive':ar.get('file','enhancing-data.zip'),'json_file':entry['json_file'],
                    'mapping_key':entry['mapping_key'],'license':source.get('license',source.get('metadata',{}).get('license'))}
                if digest not in catalog:catalog[digest]={**entry,'sources':[]}
                catalog[digest]['sources'].append(provenance)
    with (a.out/'catalog.jsonl').open('w',encoding='utf-8') as f:
        for digest,entry in sorted(catalog.items()):
            entry['record_id']=digest[:16]
            entry['split']='validation' if int(digest[:8],16)%100<5 else 'train'
            entry['split_basis']='implementation-assumption: deterministic SHA256 split, not original paper split'
            f.write(json.dumps(entry,ensure_ascii=False)+'\n')
    report['catalog_unique_records']=len(catalog)
    report['catalog_split_counts']={split:sum(int(h[:8],16)%100<5 if split=='validation' else int(h[:8],16)%100>=5 for h in catalog) for split in ('train','validation')}
    (a.out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 数据审计\n\n审计日期2026-10-27，完整文件清单、CRC/哈希、字段和样例见audit.json。\n']
    for r in report['sources']:
        lines.append(f"\n## {r['source']}\n\n状态：{r['status']}。来源：{r['url']}。\n")
        if 'revision' in r:lines.append(f"revision：{r['revision']}；许可：{r['license']}；gated：{r['gated']}。\n")
        ars=r.get('archives',[])+([r['archive']] if 'archive' in r else [])
        for ar in ars:lines.append(f"\n- {ar.get('file','Zenodo archive')}: status={ar.get('status',r['status'])}, images={ar.get('image_count','unverified')}, JSON records={ar.get('json_record_count','unverified')}, dialogue-like={ar.get('dialogue_like_record_count','unverified')}\n")
    lines.append(f"\n去重后的可核对标注记录：{len(catalog)}；canonical catalog见catalog.jsonl，含每条record hash、原始字段、Q/A、来源和实现假设split。\n")
    lines.append('\n标注记录与JSON文件数必须区分；questionN/answerN是实际多轮Q/A结构，已排除占位回答。不能将图像数、Q/A条目数直接认作论文1673个JSON或其原始训练split。未执行任何打包程序，未打开pickle。\n')
    (a.out/'report.md').write_text(''.join(lines),encoding='utf-8')
    print('DATA AUDIT '+str(a.out),flush=True)

if __name__=='__main__':main()
