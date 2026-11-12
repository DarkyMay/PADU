"""Evaluate a fixed PaDU monitor against a COCO-pretrained RT-DETRv2-R50 detector."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_DISABLE_TELEMETRY='1')
from pathlib import Path
import sys,json,csv,time,hashlib
R=Path.cwd(); O=R; D=O/'runs/C3/rtdetrv2_r50_coco';D.mkdir(parents=True,exist_ok=True)
os.environ['HF_HOME']=str(O/'model_cache/huggingface')
os.environ['XDG_CACHE_HOME']=str(O/'model_cache')
REV='282494075698cab9faa1096ae26856890030c817'
MODEL='PekingU/rtdetr_v2_r50vd'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def run():
    import numpy as np,torch
    from PIL import Image
    from transformers import RTDetrImageProcessor,RTDetrV2ForObjectDetection
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score,average_precision_score
    from scipy.stats import spearmanr
    assert torch.cuda.is_available()
    weights=O/'models/rtdetrv2_r50_coco';weights.mkdir(parents=True,exist_ok=True)
    required = ['config.json', 'preprocessor_config.json', 'model.safetensors']
    missing = [name for name in required if not (weights / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {missing} under {weights}; see docs/DATA.md')
    assert digest(weights/'model.safetensors')=='3331d977dbc0c7a6cdae9ec0b0b6ad156eb6720d65b7cf0fa710dcc541d88d71','R50 weights hash mismatch'
    manifest={'repo':MODEL,'revision':REV,'files':{},'protocol':{'training':'COCO pretrained, no XWOD fine-tuning in this arm','detector_pretraining_confound':True,'imgsz':[640,640],'conf':0.25,'nms':False,'dtype':'float32','score':'unchanged class-aware greedy recall * matched-IoU','PaDU':'unchanged YOLOv8m-supervised checkpoints; archived CPU predictions','thresholds':'primary original A train thresholds; secondary B train tertiles; never test-fit'}}
    for name in ['config.json','preprocessor_config.json','model.safetensors']:
        target=weights/name
        manifest['files'][name]={'sha256':digest(target),'bytes':target.stat().st_size}
    model=RTDetrV2ForObjectDetection.from_pretrained(weights,local_files_only=True).to('cuda').eval()
    processor=RTDetrImageProcessor.from_pretrained(weights,local_files_only=True)
    manifest['parameters']=sum(p.numel() for p in model.parameters())
    manifest['versions']={'torch':torch.__version__,'transformers':__import__('transformers').__version__}
    manifest['processor']=processor.to_dict()
    names=['person','car','truck','motorcycle','bus','bicycle']
    aliases={'motorbike':'motorcycle','bike':'bicycle'}
    mapping={int(k):names.index(aliases.get(v,v)) for k,v in model.config.id2label.items() if aliases.get(v,v) in names}
    assert set(mapping.values())==set(range(6)),mapping
    manifest['class_map']={str(k):{'coco_name':model.config.id2label[k],'xwod_index':v} for k,v in mapping.items()}
    (D/'manifest.json').write_text(json.dumps(manifest,indent=2))
    sys.path.insert(0,str(R/'core'))
    from gen_detectability_labels import load_gt,score_image
    scores={}
    for split,part in [('train','train'),('val','valid'),('test','test')]:
        target=D/(split+'_frames.jsonl')
        done={}
        if target.exists():
            for line in target.read_text().splitlines():
                row=json.loads(line);done[row['frame_id']]=row
        files=sorted((R/'data/xwod'/part/'images').glob('*.jpg'))
        with target.open('a') as dst,torch.inference_mode():
            for i,path in enumerate(files):
                if path.name in done: continue
                with Image.open(path) as src: image=src.convert('RGB')
                inputs=processor(images=image,return_tensors='pt').to('cuda')
                prediction=model(**inputs)
                output=processor.post_process_object_detection(prediction,target_sizes=torch.tensor([(image.height,image.width)]),threshold=.25)[0]
                labels=output['labels'].cpu().tolist();keep=[i for i,k in enumerate(labels) if k in mapping]
                boxes=output['boxes'].cpu().tolist();conf=output['scores'].cpu().tolist()
                pb=[boxes[i] for i in keep];pc=[mapping[labels[i]] for i in keep]
                gt=load_gt(str(R/'data/xwod'/part/'labels'/(path.stem+'.txt')))
                result=score_image(gt,pb,pc,image.width,image.height)
                row={'frame_id':path.name,'n_gt':len(gt),'n_pred':len(pb),'recall':None,'localization':None,'q':None,'boxes':pb,'classes':pc,'conf':[conf[i] for i in keep]}
                if result is not None:row.update(recall=result[0],localization=result[1],q=result[2])
                dst.write(json.dumps(row)+'\n');dst.flush();done[path.name]=row
                if (i+1)%100==0: print(split,i+1,len(files),flush=True)
        scores[split]=done
    train=np.array([v['q'] for v in scores['train'].values() if v['q'] is not None])
    original=json.loads((R/'data/understanding_labels/detectability_thresholds.json').read_text())
    q1,q2=np.quantile(train,[1/3,2/3]).tolist()
    thresholds={'original_A_train':original,'B_train_tertiles':{'q1':q1,'q2':q2}}
    (D/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
    with (O/'runs/A1_A2/frame_predictions.csv').open() as f: rows=list(csv.DictReader(f))
    summary={'manifest':manifest,'thresholds':thresholds,'metrics':[]}
    for label,thr in thresholds.items():
        for seed in range(3):
            cohort=[r for r in rows if int(r['seed'])==seed and scores['test'].get(r['frame_id'],{}).get('q') is not None]
            q=np.array([scores['test'][r['frame_id']]['q'] for r in cohort]);event=q<=thr['q1'];classes=np.where(q<=thr['q1'],2,np.where(q<=thr['q2'],1,0))
            ph=np.array([float(r['p_det_2']) for r in cohort]);pred=np.array([int(r['pred_detect']) for r in cohort])
            m={'seed':seed,'threshold_definition':label,'n':len(cohort),'n_hard':int(event.sum()),'balanced_accuracy':float(balanced_accuracy_score(classes,pred)),'spearman_risk_vs_negative_q':float(spearmanr(ph,-q).statistic)}
            if 0<event.sum()<len(event):m.update(auroc=float(roc_auc_score(event,ph)),ap=float(average_precision_score(event,ph)))
            summary['metrics'].append(m)
    (D/'summary.json').write_text(json.dumps(summary,indent=2))
    (D/'SUCCESS.json').write_text(json.dumps({'completed':time.time(),'frames':{k:len(v) for k,v in scores.items()}}))
    print('C3_PRETRAINED_ARM_COMPLETE',flush=True)

if __name__ == '__main__':
    run()
