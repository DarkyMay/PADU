"""Fine-tune RT-DETRv2-R50 on XWOD using the supplied configuration."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import argparse,json,random,time,hashlib,math
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import RTDetrImageProcessor,RTDetrV2ForObjectDetection
from data import Images,Collate,category_mapping,to_device

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    ap.add_argument('--out-dir',type=Path);ap.add_argument('--seed',type=int)
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args();c=json.loads(a.config.read_text())
    if a.out_dir:c['output_dir']=str(a.out_dir)
    if a.seed is not None:c['seed']=a.seed
    print(json.dumps(c,indent=2),flush=True)
    if a.dry_run:return
    assert torch.cuda.is_available()
    out=Path(c['output_dir']).resolve();assert Path.cwd() in out.parents
    assert not out.exists(),'Use a new --out-dir; existing runs are protected'
    out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(c,indent=2))
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model=RTDetrV2ForObjectDetection.from_pretrained(c['model_dir'],local_files_only=True).cuda()
    processor=RTDetrImageProcessor.from_pretrained(c['model_dir'],local_files_only=True)
    mapping=category_mapping(model.config,c['label_names'])
    label_root=Path(c['validation_label_root']);token_root=Path(c['validation_token_root'])
    u=json.loads((label_root/'val_understanding_labels.json').read_text());d=json.loads((label_root/'val_detectability_labels.json').read_text())
    with np.load(token_root/'val_meta.npz',allow_pickle=True) as meta:
        allowed={str(f) for f,v in zip(meta['filenames'],meta['valid']) if v and str(f) in u and str(f) in d}
    assert len(allowed)==1740,'Original PaDU validation cohort changed; inspect before training'
    ds=Images(c['dataset_root'],'train',mapping,c['horizontal_flip_probability']);va=Images(c['dataset_root'],'val',mapping,allowed_filenames=allowed)
    loaders=[DataLoader(d,batch_size=c['batch_size'],shuffle=i==0,num_workers=c['workers'],collate_fn=Collate(processor),drop_last=i==0) for i,d in enumerate([ds,va])]
    groups=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        lr=c['backbone_learning_rate'] if 'backbone' in name and 'norm' not in name else c['learning_rate']
        wd=0 if ('encoder' in name or 'decoder' in name) and ('norm' in name or 'bn' in name) else c['weight_decay']
        groups.append({'params':[p],'lr':lr,'initial_lr':lr,'weight_decay':wd})
    opt=torch.optim.AdamW(groups,betas=tuple(c['betas']))
    scaler=torch.amp.GradScaler('cuda')
    manifest={'n_parameters':sum(p.numel() for p in model.parameters()),'class_mapping':mapping,'torch':torch.__version__,'transformers':__import__('transformers').__version__,'device':torch.cuda.get_device_name(0),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'train_frames':len(ds),'val_frames':len(va),'train_files':[p.name for p in ds.files],'val_files':[p.name for p in va.files],'excluded_val_files':va.excluded,'validation_cohort_policy':c['validation_cohort_policy']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    step=0;best=float('inf');bad=0
    with (out/'history.jsonl').open('w') as history:
        for ep in range(c['epochs']):
            model.train();opt.zero_grad(set_to_none=True);total=0;n=0;t=time.time()
            for index,(batch,names) in enumerate(loaders[0]):
                group_start=index//c['gradient_accumulation']*c['gradient_accumulation']
                group_size=min(c['gradient_accumulation'],len(loaders[0])-group_start)
                batch=to_device(batch,'cuda')
                with torch.autocast('cuda',dtype=torch.float16):loss=model(**batch).loss
                assert torch.isfinite(loss),'Nonfinite training loss'
                scaler.scale(loss/group_size).backward();total+=float(loss.detach())*len(names);n+=len(names)
                if index==0 or (index+1)%50==0:
                    print(json.dumps({'epoch':ep+1,'phase':'train','batch':index+1,'batches':len(loaders[0]),'loss':float(loss.detach()),'optimizer_steps':step}),flush=True)
                if (index+1)%c['gradient_accumulation']==0 or index+1==len(loaders[0]):
                    scale=min(1.,(step+1)/c['warmup_optimizer_steps'])
                    for g in opt.param_groups:g['lr']=g['initial_lr']*scale
                    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'])
                    scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);step+=1
            model.eval();vl=0;vn=0
            with torch.inference_mode():
                for index,(batch,names) in enumerate(loaders[1]):
                    value=model(**to_device(batch,'cuda')).loss
                    assert torch.isfinite(value),'Nonfinite validation loss'
                    vl+=float(value)*len(names);vn+=len(names)
                    if index==0 or (index+1)%100==0:
                        print(json.dumps({'epoch':ep+1,'phase':'val','batch':index+1,'batches':len(loaders[1]),'loss':float(value)}),flush=True)
            vl/=vn
            row={'epoch':ep+1,'train_loss':total/n,'val_loss':vl,'optimizer_steps':step,'seconds':time.time()-t}
            history.write(json.dumps(row)+'\n');history.flush();print(json.dumps(row),flush=True)
            improved=vl<best
            if improved:best=vl;bad=0
            else:bad+=1
            bundle={'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':c,'epoch':ep+1,'validation_loss':vl,'best_validation_loss':best,'optimizer_state':opt.state_dict(),'scaler_state':scaler.state_dict(),'model_config':model.config.to_dict()}
            temp=out/'last.pt.tmp';torch.save(bundle,temp);temp.replace(out/'last.pt')
            if improved:
                import shutil;shutil.copy2(out/'last.pt',out/'best.pt')
            if bad>=c['patience']:break
    (out/'SUCCESS.json').write_text(json.dumps({'best_validation_loss':best,'completed':time.time()}))
if __name__=='__main__':main()
