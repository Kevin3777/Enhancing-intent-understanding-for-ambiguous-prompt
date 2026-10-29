"""Download only pinned public weights, atomically; no token/key is recorded."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, json, shutil, time, urllib.request

MODELS = {
    'sd15': ('stable-diffusion-v1-5/stable-diffusion-v1-5', '451f4fe16113bff5a5d2269ed5ad43b0592e9a14', [
        'model_index.json', 'scheduler/scheduler_config.json', 'text_encoder/config.json',
        'text_encoder/model.fp16.safetensors', 'tokenizer/merges.txt', 'tokenizer/vocab.json',
        'tokenizer/tokenizer_config.json', 'tokenizer/special_tokens_map.json', 'unet/config.json',
        'unet/diffusion_pytorch_model.fp16.safetensors', 'vae/config.json',
        'vae/diffusion_pytorch_model.fp16.safetensors', 'README.md']),
    'clip': ('openai/clip-vit-base-patch32', '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268', [
        'config.json', 'pytorch_model.bin', 'preprocessor_config.json', 'merges.txt', 'vocab.json',
        'special_tokens_map.json', 'tokenizer_config.json', 'tokenizer.json', 'README.md'])}

def download(url, dest, expected=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        sha = hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest()
        if expected is None or sha == expected:
            return {'path': str(dest), 'sha256': sha, 'bytes': dest.stat().st_size, 'cached': True}
        raise ValueError(f'Existing file hash mismatch, refusing overwrite: {dest}')
    part = dest.with_suffix(dest.suffix + '.part')
    req = urllib.request.Request(url, headers={'User-Agent': 'VCA-reproduction-audit/1.0'})
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(req, timeout=60) as resp, part.open('wb') as out:
                size = int(resp.headers.get('Content-Length', 0))
                if size and shutil.disk_usage(dest.parent).free < size + 700_000_000:
                    raise RuntimeError(f'Insufficient disk for {dest}')
                while data := resp.read(4*1024*1024):
                    digest.update(data)
                    out.write(data)
            sha = digest.hexdigest()
            if expected and sha != expected:
                raise ValueError(f'SHA256 mismatch: {dest}')
            part.replace(dest)
            print(f'DOWNLOADED {dest.name} {dest.stat().st_size}', flush=True)
            return {'path':str(dest),'sha256':sha,'bytes':dest.stat().st_size,'url':url}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1+attempt)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    project_root=a.root.resolve()
    manifest={}
    for name,(repo,revision,files) in MODELS.items():
        info=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',timeout=30))
        by_name={x['rfilename']:x for x in info['siblings']}
        target=project_root/'models'/name
        def one(filename):
            expected=by_name.get(filename,{}).get('lfs',{}).get('sha256')
            row=download(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}',target/filename,expected)
            row['path']=(target/filename).relative_to(project_root).as_posix()
            return row
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(one,files))
        manifest[name]={'model_id':repo,'revision':revision,'files':results,'license':info.get('cardData',{}).get('license','see model card')}
        (target/'vca_manifest.json').write_text(json.dumps(manifest[name],indent=2),encoding='utf-8')
    (project_root/'docs/assets_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('ALL PINNED ASSETS VERIFIED',flush=True)

if __name__=='__main__':main()
