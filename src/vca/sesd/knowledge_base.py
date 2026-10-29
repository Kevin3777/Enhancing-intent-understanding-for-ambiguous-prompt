from pathlib import Path
import json

class KnowledgeBase:
    def __init__(self,path=None):
        path=path or Path(__file__).resolve().parents[3]/'data/knowledge.json'
        self.entries=json.loads(Path(path).read_text(encoding='utf-8'));self.version=0
        if len({r['id'] for r in self.entries})!=len(self.entries):raise ValueError('Duplicate knowledge id')

    def append_session(self,prompt,feedback,session_id,round_id,image_path=None,clip_feedback=None):
        self.version+=1
        entry={'id':f'session:{session_id}:{round_id}','category':'Session History','term':prompt,
            'sense':prompt,'aliases':[],'text':f'{prompt} | user feedback: {feedback}',
            'source':f'user-session:{session_id}:round:{round_id}',
            'classification':'implementation-assumption: session memory, not external verified fact',
            'image_path':image_path,'clip_feedback':clip_feedback}
        self.entries.append(entry)
        return entry
