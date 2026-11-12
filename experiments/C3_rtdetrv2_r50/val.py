"""Evaluate a pretrained or fine-tuned RT-DETRv2-R50 detector on XWOD."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse,json,hashlib,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import RTDetrImageProcessor,RTDetrV2ForObjectDetection
from data import category_mapping
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--split',choices=['train','val','test'],default='test')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--inspect-only',action='store_true')
    a=ap.parse_args();c=json.loads(a.config.read_text())
    model=RTDetrV2ForObjectDetection.from_pretrained(c['model_dir'],local_files_only=True)
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model_state'],strict=True);c=ck['config']
    model=model.to(a.device).eval();processor=RTDetrImageProcessor.from_pretrained(c['model_dir'],local_files_only=True)
    mapping={v:k for k,v in category_mapping(model.config,c['label_names']).items()}
    info={'parameters':sum(p.numel() for p in model.parameters()),'configuration':c,'class_mapping':mapping,'checkpoint':str(a.checkpoint),'device':a.device,'split':a.split}
    if a.checkpoint:info['checkpoint_sha256']=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    print(json.dumps(info,indent=2),flush=True)
    if a.inspect_only:return
    out=a.output_dir.resolve();assert Path.cwd() in out.parents and not out.exists()
    out.mkdir(parents=True);(out/'manifest.json').write_text(json.dumps(info,indent=2))
    sys.path.insert(0,'core')
    from gen_detectability_labels import load_gt,score_image
    part=Path(c['dataset_root'])/('valid' if a.split=='val' else a.split);rows=[]
    with (out/'frames.jsonl').open('w') as dst,torch.inference_mode():
        files=sorted((part/'images').glob('*.jpg'))
        for i,p in enumerate(files):
            with Image.open(p) as src:image=src.convert('RGB')
            result=model(**processor(images=image,return_tensors='pt').to(a.device))
            result=processor.post_process_object_detection(result,target_sizes=torch.tensor([(image.height,image.width)]),threshold=c['inference_confidence'])[0]
            labels=result['labels'].cpu().tolist();boxes=result['boxes'].cpu().tolist();keep=[j for j,k in enumerate(labels) if k in mapping]
            pb=[boxes[j] for j in keep];pc=[mapping[labels[j]] for j in keep]
            gt=load_gt(str(part/'labels'/(p.stem+'.txt')));score=score_image(gt,pb,pc,image.width,image.height,c['iou_match_threshold'])
            row={'frame_id':p.name,'n_gt':len(gt),'n_pred':len(pb),'q':score[2] if score else None,'recall':score[0] if score else None,'localization':score[1] if score else None}
            dst.write(json.dumps(row)+'\n');dst.flush();rows.append(row)
            if (i+1)%100==0:print('EVALUATED',i+1,len(files),flush=True)
    summary={'n_frames':len(rows),'n_scored':sum(r['q'] is not None for r in rows),'mean_q':float(np.mean([r['q'] for r in rows if r['q'] is not None])),'completed':time.time()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'SUCCESS.json').write_text(json.dumps(summary))
if __name__=='__main__':main()
