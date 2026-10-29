from pathlib import Path
import json,uuid
from .logging import write_json

class Session:
    def __init__(self,folder,prompt=None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.file=self.folder/'session.json'
        if self.file.exists():
            self.data=json.loads(self.file.read_text(encoding='utf-8'))
            if prompt is not None and prompt!=self.data['original_prompt']:raise ValueError('Session prompt mismatch')
        else:
            if prompt is None:raise ValueError('A new session requires an initial prompt')
            self.data={'id':str(uuid.uuid4()),'original_prompt':prompt,'current_prompt':prompt,'rounds':[],
                       'rolled_back_rounds':[],'next_round_id':1,'stop_reason':None,'human_ratings':[],
                       'classification':'partial_experiment_replication'}
            self.save()
    def save(self):write_json(self.file,self.data)
    def append(self,row):
        self.data['rounds'].append(row);self.data['current_prompt']=row['new_prompt']
        self.data['current_image']=row['image'];self.data['next_round_id']=row['round_id']+1;self.save()
    def undo(self):
        if not self.data['rounds']:raise ValueError('No generated rounds to undo')
        row=self.data['rounds'].pop();self.data['rolled_back_rounds'].append(row)
        snapshot=row['before_snapshot']
        self.data.update(snapshot);self.data['stop_reason']=None;self.save()
        return row
