"""Fetch a new, pinned local LLM; preserve all existing project models."""
from pathlib import Path
import argparse,hashlib,json,time,urllib.request

REPO='Qwen/Qwen2.5-0.5B-Instruct'
REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
FILES={'config.json':None,'generation_config.json':None,'merges.txt':None,'tokenizer.json':None,
       'tokenizer_config.json':None,'vocab.json':None,'LICENSE':None,
       'model.safetensors':'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);manifest=[]
    for name,expected in FILES.items():
        path=a.out/name
        if path.exists():
            digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
            if expected and digest!=expected:raise ValueError('Existing model file differs; preserved: '+str(path))
            manifest.append({'file':name,'sha256':digest,'bytes':path.stat().st_size,'cached':True});continue
        url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        part=path.with_suffix(path.suffix+'.part')
        for attempt in range(5):
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'VCA-local-llm-verification/1.0'})
                h=hashlib.sha256()
                with urllib.request.urlopen(request,timeout=60) as response,part.open('wb') as f:
                    while chunk:=response.read(4*1024*1024):f.write(chunk);h.update(chunk)
                digest=h.hexdigest()
                if expected and expected!=digest:raise ValueError(f'Hash mismatch for {name}: {digest}')
                part.rename(path)
                manifest.append({'file':name,'sha256':digest,'bytes':path.stat().st_size,'url':url})
                print('VERIFIED '+name,flush=True);break
            except (OSError,ValueError) as exc:
                print('RETRY',name,attempt,type(exc).__name__,flush=True)
                if attempt==4:raise
                time.sleep(1+attempt)
    dest=a.out/'vca_manifest.json'
    data={'id':REPO,'revision':REVISION,'license':'apache-2.0','files':manifest,
          'role':'local LLM engineering substitute; not original ChatGPT-4 or paper checkpoint'}
    if dest.exists():
        old=json.loads(dest.read_text(encoding='utf-8'))
        if old['id']!=REPO or old['revision']!=REVISION:raise ValueError('Existing manifest mismatch; preserved')
    else:dest.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('LOCAL_LLM_READY '+str(a.out),flush=True)

if __name__=='__main__':main()
