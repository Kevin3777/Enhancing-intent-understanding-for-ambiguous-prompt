from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import argparse,json,hashlib
from vca.runtime.logging import write_json

def display_path(path):
    try:return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:return str(Path(path).resolve())

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'outputs/dataset_audit_final/catalog.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'data/catalog.jsonl');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing catalog; choose another output path')
    rows=[json.loads(line) for line in a.input.read_text(encoding='utf-8').splitlines()]
    rows.sort(key=lambda r:r['record_sha256']);nval=int(.05*len(rows))
    for i,row in enumerate(rows):
        row['split']='validation' if i<nval else 'train'
        row['split_basis']='implementation-assumption: SHA256 order, floor(5% N) validation; not original paper split'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary={'source':display_path(a.input),'source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
        'output':display_path(a.output),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
        'records':len(rows),'train':len(rows)-nval,'validation':nval,'method_basis':'PDF p9 95/5 ratio; assignment is implementation-assumption'}
    write_json(a.output.with_suffix('.manifest.json'),summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
