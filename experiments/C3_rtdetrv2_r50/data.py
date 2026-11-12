import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def load_yolo_boxes(label):
    """Read validated native YOLO box rows; ambiguous mixed files are rejected."""
    rows=[]
    for number,line in enumerate(Path(label).read_text().splitlines(),1):
        if not line.strip():continue
        values=np.asarray([float(x) for x in line.split()],dtype=np.float32)
        assert np.isfinite(values).all(),f'Nonfinite annotation {label}:{number}'
        assert len(values)==5,f'Unresolved non-box annotation {label}:{number}; do not infer its meaning'
        rows.append(values)
    return np.stack(rows) if rows else np.empty((0,5),dtype=np.float32)

def category_mapping(config,names):
    aliases={'motorbike':'motorcycle','bike':'bicycle'}
    reverse={aliases.get(v,v):int(k) for k,v in config.id2label.items()}
    return {i:reverse[n] for i,n in enumerate(names)}

class Images(Dataset):
    def __init__(self,root,split,mapping,flip=0,allowed_filenames=None):
        self.root=Path(root)/('valid' if split=='val' else split)
        self.files=sorted((self.root/'images').glob('*.jpg'))
        self.excluded=[]
        if allowed_filenames is not None:
            self.excluded=[p.name for p in self.files if p.name not in allowed_filenames]
            self.files=[p for p in self.files if p.name in allowed_filenames]
            assert {p.name for p in self.files}==set(allowed_filenames)
        self.mapping=mapping;self.flip=flip
        assert self.files
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        path=self.files[i]
        with Image.open(path) as src:im=src.convert('RGB')
        width,height=im.size;label=self.root/'labels'/(path.stem+'.txt')
        assert label.exists(),'Missing annotation '+str(label)
        gt=load_yolo_boxes(label)
        flip=random.random()<self.flip
        if flip:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        annotations=[]
        for cls,cx,cy,w,h in gt:
            x=(cx-w/2)*width;y=(cy-h/2)*height;bw=w*width;bh=h*height
            if flip:x=width-x-bw
            assert int(cls) in self.mapping
            annotations.append({'image_id':i,'category_id':self.mapping[int(cls)],'bbox':[float(x),float(y),float(bw),float(bh)],'area':float(bw*bh),'iscrowd':0})
        return im,{'image_id':i,'annotations':annotations},path.name

class Collate:
    def __init__(self,processor):self.processor=processor
    def __call__(self,batch):
        images,labels,names=zip(*batch)
        out=self.processor(images=list(images),annotations=list(labels),return_tensors='pt')
        return dict(out),list(names)

def to_device(batch,device):
    return {k:[{kk:vv.to(device) if isinstance(vv,torch.Tensor) else vv for kk,vv in x.items()} for x in v] if k=='labels' else v.to(device) for k,v in batch.items()}
