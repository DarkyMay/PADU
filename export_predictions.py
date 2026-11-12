"""Export CPU predictions from three validation-selected PaDU checkpoints."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse, csv, gc, hashlib, json, sys, time, resource
from pathlib import Path
sys.dont_write_bytecode = True

ROOT=Path.cwd()
OUT=ROOT
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(name,data):
 p=OUT/name
 assert OUT.resolve() in p.resolve().parents
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--pilot',action='store_true')
 ap.add_argument('--batch',type=int,default=2)
 args=ap.parse_args()
 assert OUT.resolve()==Path.cwd()
 print('START',time.strftime('%Y-%m-%d %H:%M:%S'), 'device=cpu, threads=1, batch=',args.batch,flush=True)
 import numpy as np
 import torch
 torch.set_num_threads(1)
 torch.set_num_interop_threads(1)
 sys.path.insert(0,str(ROOT/'core'))
 from unified_model import UnifiedWeatherModel,WeatherUnderstandingDataset
 ds=WeatherUnderstandingDataset('test',str(ROOT/'data/weather_patch_tokens'),str(ROOT/'data/understanding_labels'))
 assert len(ds.items)==len(set(ds.items))
 assert len(ds.items)==2320, ('Unexpected cohort',len(ds.items))
 token_file=ROOT/'data/weather_patch_tokens/test_tokens.npy'
 manifest={'time_started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'python':sys.version,
 'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':1,'batch':args.batch,
 'training_performed':False,'split':'test','n_frames':len(ds.items),
 'model_source_sha256':sha(ROOT/'core/unified_model.py'),
 'exporter_sha256':sha(Path(__file__)),
 'token_cache':{'path':str(token_file),'size':token_file.stat().st_size,'shape':list(ds.tokens.shape),'dtype':str(ds.tokens.dtype),'sha256':None,'hash_status':'not computed: multi-GB original cache read-only; meta and checkpoints hashed'},
 'input_sha256':{str(p):sha(p) for p in [ROOT/'data/weather_patch_tokens/test_meta.npz',ROOT/'data/understanding_labels/test_understanding_labels.json',ROOT/'data/understanding_labels/test_detectability_labels.json',ROOT/'data/understanding_labels/detectability_thresholds.json']},
 'excluded_frames':[str(f) for f in ds.fnames_tok if f not in set(ds.items)],
 'checkpoints':[], 'cgroup':{}}
 for n in ['cpu.max','memory.max','memory.current']:
  p=Path('/sys/fs/cgroup')/n
  if p.exists():manifest['cgroup'][n]=p.read_text().strip()
 models=[]
 for seed in range(3):
  ckpath=ROOT/f'runs/weatherfilm_runs/uwm_main_seed{seed}/best_uwm.pt'
  ck=torch.load(ckpath,map_location='cpu',weights_only=False)
  ma=ck.get('args',{})
  model=UnifiedWeatherModel(d_model=ma.get('d_model',384),n_layers=ma.get('n_layers',3),n_heads=ma.get('n_heads',6),dropout=ma.get('dropout',0.1)).float().eval()
  model.load_state_dict(ck['model_state'],strict=True)
  assert all(p.device.type=='cpu' for p in model.parameters())
  manifest['checkpoints'].append({'seed':seed,'path':str(ckpath),'sha256':sha(ckpath),'args':ma,'epoch':ck.get('epoch'),'val_metrics':ck.get('val_metrics'),'n_parameters':sum(p.numel() for p in model.parameters())})
  models.append(model)
  del ck;gc.collect()
  print('LOADED seed',seed,'epoch',manifest['checkpoints'][-1]['epoch'],flush=True)
 # Numerical smoke check: eval-mode output must agree for batch=1 and batch=2.
 xt=torch.from_numpy(np.stack([np.array(ds.tokens[ds.f2tok[f]],dtype=np.float32,copy=True) for f in ds.items[:2]]))
 with torch.inference_mode():
  batch_o=models[0](xt)
  single_o=models[0](xt[:1])
 maxdiff={k:float((v[:1]-single_o[k]).abs().max()) for k,v in batch_o.items()}
 assert max(maxdiff.values())<1e-4,maxdiff
 manifest['batch_consistency_max_abs_difference']=maxdiff
 del xt,batch_o,single_o;gc.collect()
 if args.pilot:
  t=time.monotonic()
  with torch.inference_mode():
   for start in range(0,8,args.batch):
    x=torch.from_numpy(np.stack([np.array(ds.tokens[ds.f2tok[f]],dtype=np.float32,copy=True) for f in ds.items[start:start+args.batch]]))
    for model in models:
     out=model(x)
     assert all(torch.isfinite(v).all() for v in out.values())
  manifest['pilot_wall_seconds']=time.monotonic()-t
  manifest['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  manifest['pilot_frames']=8
  write_json('pilot.json',manifest)
  print('PILOT_DONE',manifest['pilot_wall_seconds'],'seconds;',manifest['max_rss_kib'],'KiB maxRSS',flush=True)
  return
 resultdir=OUT/'runs/A1_A2';resultdir.mkdir(parents=True,exist_ok=True)
 target=resultdir/'frame_predictions.csv'
 assert not target.exists(), 'Complete output exists; inspect before replacing.'
 partial=target.with_suffix('.partial.csv')
 fields=['frame_id','seed','token_row','weather','true_severity','true_detect','true_side','true_count','true_n_objects','true_n_small','true_q','true_recall','true_localization','pred_severity','pred_detect','pred_side','pred_count','pred_n_objects','pred_n_small','risk','p_sev_0','p_sev_1','p_sev_2','p_det_0','p_det_1','p_det_2','logit_det_0','logit_det_1','logit_det_2']
 t=time.monotonic()
 with partial.open('w',newline='',encoding='utf-8') as f,torch.inference_mode():
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for start in range(0,len(ds.items),args.batch):
   names=ds.items[start:start+args.batch]
   x=torch.from_numpy(np.stack([np.array(ds.tokens[ds.f2tok[n]],dtype=np.float32,copy=True) for n in names]))
   for seed,model in enumerate(models):
    output=model(x)
    assert all(torch.isfinite(v).all() for v in output.values())
    ps=output['severity'].softmax(1).numpy();pd=output['detect'].softmax(1).numpy()
    sp=output['severity'].argmax(1).numpy();dp=output['detect'].argmax(1).numpy();side=output['side'].argmax(1).numpy()
    counts=output['count'].expm1().clamp(min=0).numpy();density=output['density'].expm1().clamp(min=0).numpy()
    logits=output['detect'].numpy()
    for j,n in enumerate(names):
     u,d=ds.u[n],ds.d[n];raw=float(dp[j])+0.5*float(sp[j]);sl=u['vqa']['largest_side']
     row={'frame_id':n,'seed':seed,'token_row':ds.f2tok[n],'weather':u['weather'],
      'true_severity':int(u['severity']),'true_detect':int(d['difficulty']),'true_side':0 if sl=='left' else (1 if sl=='right' else -1),
      'true_count':u['vqa']['count_total'],'true_n_objects':u['density']['n_objects'],'true_n_small':u['density']['n_small'],
      'true_q':d['score'],'true_recall':d['recall'],'true_localization':d['loc_qual'],
      'pred_severity':int(sp[j]),'pred_detect':int(dp[j]),'pred_side':int(side[j]),
      'pred_count':float(counts[j]),'pred_n_objects':float(density[j,0]),'pred_n_small':float(density[j,1]),
      'risk':0 if raw<=.5 else (1 if raw<=1.75 else 2)}
     row.update({f'p_sev_{k}':float(ps[j,k]) for k in range(3)})
     row.update({f'p_det_{k}':float(pd[j,k]) for k in range(3)})
     row.update({f'logit_det_{k}':float(logits[j,k]) for k in range(3)})
     writer.writerow(row)
   if start%100==0 or start+args.batch>=len(ds.items):
    f.flush();elapsed=time.monotonic()-t
    print('PROGRESS',min(start+args.batch,len(ds.items)),'/',len(ds.items),'frames x3 seeds;',round(elapsed,1),'s; maxRSS_KiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    write_json('progress.json',{'processed_frames':min(start+args.batch,len(ds.items)),'total':len(ds.items),'seeds':3,'elapsed_seconds':elapsed,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    # Release read-only mapped pages; no writes to token files.
    if hasattr(ds.tokens._mmap,'madvise'):ds.tokens._mmap.madvise(4)
   del x,output
 partial.rename(target)
 manifest['elapsed_seconds']=time.monotonic()-t
 manifest['prediction_sha256']=sha(target)
 manifest['prediction_rows']=len(ds.items)*3
 manifest['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 manifest['time_completed']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
 write_json('runs/A1_A2/inference_manifest.json',manifest)
 print('EXPORT_DONE',manifest['prediction_rows'],'rows',manifest['prediction_sha256'],flush=True)

if __name__=='__main__':main()
