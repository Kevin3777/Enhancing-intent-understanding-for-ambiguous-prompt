from pathlib import Path
import datetime,hashlib,json,platform,subprocess,sys
import torch

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8');temp.replace(path)

def environment(root):
    root=Path(root).resolve();executable=Path(sys.executable).resolve()
    try:executable=executable.relative_to(root).as_posix()
    except ValueError:executable=str(executable)
    return {'python':sys.version,'executable':str(executable),'platform':platform.platform(),
        'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'source_date':'2026-10-27','device_scope':'RTX3080 10GB local partial experiment',
        'models':{k:json.loads((root/'models'/k/'vca_manifest.json').read_text()) for k in ('sd15','clip') if (root/'models'/k/'vca_manifest.json').exists()},
        'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src','scripts','configs') for p in sorted((root/folder).rglob('*'))
            if p.is_file() and p.suffix in ('.py','.yaml') and '__pycache__' not in p.parts}}
