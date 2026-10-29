"""Read-only dataset schema normalization. Archives and executables are never run."""
from pathlib import PurePosixPath
import hashlib,io,json,re,zipfile

def normalized_records(data):
    if isinstance(data,list):return [(str(i),x) for i,x in enumerate(data) if isinstance(x,dict)]
    if not isinstance(data,dict):return []
    if any(k in data for k in ('prompt','initial_prompt','turns','question1','image_path')):return [('',data)]
    for key in ('data','records','annotations'):
        if key in data and isinstance(data[key],list):return normalized_records(data[key])
    return [(str(k),v) for k,v in data.items() if isinstance(v,dict)]

def qa_pairs(record):
    pairs=[];placeholders=0
    for n in range(1,32):
        q=record.get(f'question{n}');a=record.get(f'answer{n}')
        if not isinstance(a,str) or not a.strip():continue
        if re.search(r'enter (?:the |your )?answer|please (?:enter|input)|placeholder',a,re.I):
            placeholders+=1;continue
        pairs.append({'index':n,'question':q,'answer':a,'refined_prompt':record.get(f'prompt{n}')})
    return pairs,placeholders

def record_entry(record,file,key):
    canonical=json.dumps(record,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    pairs,placeholders=qa_pairs(record)
    explicit=next((record[k] for k in ('turns','dialogues','messages','qa_pairs') if isinstance(record.get(k),list)),[])
    return {'json_file':file,'mapping_key':key,'record_sha256':hashlib.sha256(canonical).hexdigest(),
        'schema':'qa_columns' if pairs else 'explicit_turns' if explicit else 'image_prompt_or_other',
        'qa_pairs':pairs,'placeholder_answer_count':placeholders,
        'dialogue_like':len(pairs)>=2 or len(explicit)>=2,'record':record}

def audit_zip(zf,depth=0,prefix=''):
    members=[];entries=[];documents=0;images=0;skipped=[];nested=[];fields=set();json_hashes={}
    for info in zf.infolist():
        name=info.filename;path=PurePosixPath(name.replace('\\','/'));ext=path.suffix.lower()
        row={'name':name,'bytes':info.file_size,'crc32':f'{info.CRC:08x}'};members.append(row)
        if info.is_dir():continue
        if ext in ('.png','.jpg','.jpeg','.webp','.bmp'):images+=1
        if ext=='.json':
            if info.file_size>16_000_000:skipped.append({'file':name,'reason':'JSON >16MB explicit read limit'});continue
            try:
                payload=zf.read(info)  # ZipFile verifies entry CRC
                parsed=json.loads(payload.decode('utf-8-sig'));documents+=1
                row['sha256']=hashlib.sha256(payload).hexdigest();json_hashes[prefix+name]=row['sha256']
                for key,record in normalized_records(parsed):
                    fields.update(record);entries.append(record_entry(record,prefix+name,key))
            except (ValueError,UnicodeError,zipfile.BadZipFile) as e:row['parse_error']=str(e)
        if ext=='.zip':
            if depth>=2 or info.file_size>300_000_000:
                skipped.append({'file':name,'reason':'Nested archive depth/size read limit'});continue
            try:
                payload=zf.read(info);inner_sha=hashlib.sha256(payload).hexdigest()
                with zipfile.ZipFile(io.BytesIO(payload)) as child:
                    ar=audit_zip(child,depth+1,prefix+name+'!/')
                ar.update(container_file=name,container_sha256=inner_sha)
                entries.extend(ar.pop('record_entries'));nested.append(ar)
                images+=ar['image_count'];documents+=ar['json_document_count'];fields.update(ar['observed_json_fields'])
                json_hashes.update(ar['json_document_hashes'])
            except (OSError,zipfile.BadZipFile) as e:row['nested_error']=str(e)
    unique={x['record_sha256']:x for x in entries}
    missing={k:sum(k not in e['record'] for e in entries) for k in ('prompt','image_path','question1','answer1')}
    return {'members':members,'member_count':len(members),'image_count':images,
        'image_count_unit':'archive member names with image extensions, nested archives included; not dialogue count',
        'json_document_count':documents,'json_record_count':len(entries),'unique_annotation_records':len(unique),
        'dialogue_like_record_count':sum(e['dialogue_like'] for e in entries),
        'unique_dialogue_like_records':sum(e['dialogue_like'] for e in unique.values()),
        'valid_qa_pair_count':sum(len(e['qa_pairs']) for e in entries),
        'placeholder_answer_count':sum(e['placeholder_answer_count'] for e in entries),
        'observed_json_fields':sorted(fields),'missing_expected_fields':missing,
        'json_samples':[e['record'] for e in entries[:3]],'record_entries':entries,
        'json_document_hashes':json_hashes,'nested_archives':nested,'skipped':skipped,
        'dialogue_detection':'normalized list/mapping records; >=2 non-placeholder questionN/answerN pairs or explicit turns. Not assumed equal to paper editing trajectories.',
        'unsafe_paths':[r['name'] for r in members if '..' in PurePosixPath(r['name'].replace('\\','/')).parts or PurePosixPath(r['name']).is_absolute()],
        'pickle_files_not_opened':[r['name'] for r in members if PurePosixPath(r['name']).suffix.lower() in ('.pkl','.pickle','.pt','.pth')],
        'executables_not_run':[r['name'] for r in members if PurePosixPath(r['name']).suffix.lower() in ('.exe','.dll','.bat','.cmd')]}
