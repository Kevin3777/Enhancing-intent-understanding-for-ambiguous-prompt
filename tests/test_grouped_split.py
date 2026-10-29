from vca.grouped_split import grouped_split,verify_no_leak

def test_transitive_duplicate_groups_cannot_cross_split():
    def row(i,prompt,image):
        return {'record_sha256':f'{i:064x}','record':{'prompt':prompt,'image':image},'sources':[{'archive':'same.zip'}]}
    entries=[row(1,'A CAT!', 'a.png'),row(2,'a cat','b.png'),row(3,'A dog','b.png')]
    entries.extend(row(i,f'unique object {i}',f'{i}.png') for i in range(4,80))
    split=grouped_split(entries,.2);proof=verify_no_leak(entries,split)
    assert proof=={'record_overlap':0,'normalized_prompt_overlap':0,'archive_image_id_overlap':0}
    group={entries[i]['record_sha256'] for i in (0,1,2)}
    assert group<=set(split['train']) or group<=set(split['validation'])
