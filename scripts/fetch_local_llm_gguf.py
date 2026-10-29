"""Fetch a distinct pinned quantized language model, with an exact SHA-256."""
from pathlib import Path
import argparse,hashlib,json,time,urllib.request

REPO='Qwen/Qwen2.5-1.5B-Instruct-GGUF'
REVISION='91cad51170dc346986eccefdc2dd33a9da36ead9'
NAME='qwen2.5-1.5b-instruct-q4_k_m.gguf'
SHA='6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);dest=a.out/NAME
    if dest.exists():
        digest=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()
        if digest!=SHA:raise ValueError('Existing model differs; preserved')
        print('EXISTING_HASH_VERIFIED',flush=True)
    else:
        part=dest.with_suffix('.gguf.part');url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{NAME}'
        for i in range(8):
            try:
                size=part.stat().st_size if part.exists() else 0
                if size>=1117320736:raise ValueError('Partial file unexpectedly complete; preserved for audit')
                h=hashlib.sha256()
                if size:
                    with part.open('rb') as existing:
                        while chunk:=existing.read(4*1024*1024):h.update(chunk)
                req=urllib.request.Request(url,headers={'User-Agent':'VCA-local-llm-verification/1.0','Range':f'bytes={size}-'})
                with urllib.request.urlopen(req,timeout=60) as r,part.open('ab') as f:
                    content_range=r.headers.get('Content-Range','')
                    if r.status!=206 or not content_range.startswith(f'bytes {size}-'):
                        raise ValueError('Server did not honor exact resume range; partial file preserved')
                    while chunk:=r.read(4*1024*1024):f.write(chunk);h.update(chunk)
                if part.stat().st_size!=1117320736:raise OSError('Partial transfer; resume')
                if h.hexdigest()!=SHA:raise ValueError('Model hash mismatch; partial file preserved')
                part.rename(dest);break
            except (OSError,ValueError) as e:
                print('RETRY',i,type(e).__name__,flush=True)
                if i==7:raise
                time.sleep(1+i)
    path=a.out/'vca_manifest.json'
    data={'id':REPO,'revision':REVISION,'license':'apache-2.0','file':NAME,'sha256':SHA,
          'quantization':'Q4_K_M','role':'real local LLM engineering substitute, not paper ChatGPT-4'}
    if not path.exists():path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    elif json.loads(path.read_text(encoding='utf-8'))!=data:raise ValueError('Existing manifest differs; preserved')
    print('LOCAL_GGUF_READY '+str(dest),flush=True)

if __name__=='__main__':main()
