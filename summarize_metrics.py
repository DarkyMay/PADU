"""Compute seed-level metrics, risk groups and paired frame-bootstrap intervals.

Average precision uses the noninterpolated definition. Bootstrap samples
are shared across seeds and condition on the fitted models."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path.cwd()

def binary_metrics(y,p,flag=None):
 y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=np.float64)
 assert y.ndim==p.ndim==1 and y.shape==p.shape
 assert np.isin(y,[0,1]).all() and np.isfinite(p).all()
 n=len(y);pos=int(y.sum());neg=n-pos
 assert pos>0 and neg>0
 order=np.argsort(-p,kind='stable');sy=y[order];sp=p[order]
 end=np.r_[np.flatnonzero(np.diff(sp)),n-1]
 tp=np.cumsum(sy)[end];fp=1+end-tp
 tpr=np.r_[0.,tp/pos];fpr=np.r_[0.,fp/neg]
 auc=float(np.sum((tpr[1:]+tpr[:-1])*.5*np.diff(fpr)))
 recall=tp/pos;precision=tp/(1+end)
 ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
 brier=float(np.mean((p-y)**2))
 ece=0.;bins=[]
 bi=np.minimum(np.floor(p*15).astype(int),14)
 for b in range(15):
  m=bi==b
  if m.any():
   conf=float(p[m].mean());freq=float(y[m].mean());count=int(m.sum())
   ece+=count/n*abs(conf-freq)
   bins.append({'bin':b,'n':count,'mean_predicted_hard':conf,'observed_hard_fraction':freq})
 result={'n':n,'positives':pos,'prevalence':pos/n,'auroc':auc,'auprc_average_precision':ap,'binary_brier':brier,'binary_ece_15':float(ece)}
 if flag is not None:result.update(flag_metrics(y,flag))
 return result,{'roc_fpr':fpr.tolist(),'roc_tpr':tpr.tolist(),'pr_recall':recall.tolist(),'pr_precision':precision.tolist(),'score_thresholds':sp[end].tolist(),'calibration_bins':bins}

def flag_metrics(y,flag):
 y=np.asarray(y,dtype=bool);flag=np.asarray(flag,dtype=bool)
 tp=int((y&flag).sum());fp=int((~y&flag).sum());fn=int((y&~flag).sum());tn=int((~y&~flag).sum())
 return {'alert_tp':tp,'alert_fp':fp,'alert_fn':fn,'alert_tn':tn,'alert_precision':tp/(tp+fp) if tp+fp else None,'alert_recall':tp/(tp+fn) if tp+fn else None,'alert_fpr':fp/(fp+tn) if fp+tn else None,'alert_fraction':float(flag.mean()),'hard_missed_by_alert_rate':fn/(tp+fn) if tp+fn else None}

def cm(true,pred):return np.bincount(3*np.asarray(true,dtype=int)+np.asarray(pred,dtype=int),minlength=9).reshape(3,3)
def bal(matrix):return float(np.mean(np.diag(matrix)/matrix.sum(1)))
def majority(votes):
 votes=np.asarray(votes,dtype=int)
 return np.stack([(votes==k).sum(0) for k in range(3)]).argmax(0)
def risk(severity,detect):
 raw=np.asarray(detect)+.5*np.asarray(severity)
 return np.where(raw<=.5,0,np.where(raw<=1.75,1,2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonout(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def evaluate(root=ROOT,bootstrap=2000):
 source=root/'runs/A1_A2/frame_predictions.csv'
 with source.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
 index={}
 for r in rows:
  key=(r['frame_id'],int(r['seed']))
  assert key not in index;index[key]=r
 frames=[r['frame_id'] for r in rows if int(r['seed'])==0]
 assert len(frames)==2320 and len(index)==6960
 ordered=[[index[(name,s)] for name in frames] for s in range(3)]
 for field in ['true_severity','true_detect','true_side','true_q','true_recall']:
  assert all([r[field] for r in ordered[s]]==[r[field] for r in ordered[0]] for s in [1,2])
 def arr(field,dtype=float):return np.array([[r[field] for r in seed] for seed in ordered],dtype=dtype)
 true_s=arr('true_severity',int)[0];true_d=arr('true_detect',int)[0];true_side=arr('true_side',int)[0]
 ps=arr('pred_severity',int);pd=arr('pred_detect',int);side=arr('pred_side',int)
 probs=np.stack([arr(f'p_det_{k}') for k in range(3)],axis=-1)
 assert np.allclose(probs.sum(2),1,atol=1e-6) and ((probs>=0)&(probs<=1)).all()
 assert np.array_equal(probs.argmax(2),pd)
 risk_seed=risk(ps,pd);assert np.array_equal(risk_seed,arr('risk',int))
 cms_s=[cm(true_s,x) for x in ps];cms_d=[cm(true_d,x) for x in pd]
 expected_s=np.array([[879,860,277],[288,933,465],[70,657,2531]])
 expected_d=np.array([[655,455,333],[343,960,1085],[265,882,1982]])
 current_s=np.sum(cms_s,axis=0);current_d=np.sum(cms_d,axis=0)
 # Report agreement with the reference without constraining rerun predictions.
 reference_prediction_sha256='acdffd4123d8d0f064b5154a530298a52b81157ba2d2c878dba4b30f31ccc545'
 historical_comparison={'archived_severity_cm':expected_s.tolist(),'cpu_severity_cm':current_s.tolist(),'severity_count_delta':(current_s-expected_s).tolist(),'archived_detectability_cm':expected_d.tolist(),'cpu_detectability_cm':current_d.tolist(),'detectability_count_delta':(current_d-expected_d).tolist(),'exact_historical_replication':bool(np.array_equal(current_s,expected_s) and np.array_equal(current_d,expected_d)),'reference_prediction_sha256':reference_prediction_sha256,'matches_reference_predictions':sha(source)==reference_prediction_sha256,'interpretation':'Reference matrices are provided for comparison. Metrics are computed from the supplied predictions.'}
 per_seed=[]
 for seed in range(3):
  valid=true_side>=0
  row={'seed':seed,'n_frames':len(frames),'severity_acc':float((ps[seed]==true_s).mean()),'severity_balanced_accuracy':bal(cms_s[seed]),'detect_acc':float((pd[seed]==true_d).mean()),'detect_balanced_accuracy':bal(cms_d[seed]),'side_acc':float((side[seed,valid]==true_side[valid]).mean())}
  for label in ['count','n_objects','n_small']:row[label+'_mae']=float(np.abs(arr('pred_'+label)[seed]-arr('true_'+label)[seed]).mean())
  per_seed.append(row)
 summary={k:{'mean':float(np.mean([r[k] for r in per_seed])),'std_ddof0':float(np.std([r[k] for r in per_seed],ddof=0)),'values':[r[k] for r in per_seed]} for k in per_seed[0] if k not in ['seed','n_frames']}
 reference_means={'severity_acc':.624,'detect_acc':.517,'severity_balanced_accuracy':.589,'detect_balanced_accuracy':.496,'side_acc':.802}
 headline_means_reproduced=all(round(summary[key]['mean'],3)==value for key,value in reference_means.items())
 csvout(root/'runs/A1/per_seed_metrics.csv',per_seed)
 jsonout(root/'runs/A1/summary.json',{'ddof':0,'n_seeds':3,'n_frames':len(frames),'metrics':summary,'historical_rounded_headline_means_reproduced':headline_means_reproduced,'historical_comparison':historical_comparison,'confusions_per_seed':{'severity':[x.tolist() for x in cms_s],'detectability':[x.tolist() for x in cms_d]}})
 y=(true_d==2).astype(int);q=arr('true_q')[0];rec=arr('true_recall')[0]
 ms=majority(ps);md=majority(pd);mr=risk(ms,md)
 table=[];curves={}
 for seed in range(3):
  m,c=binary_metrics(y,probs[seed,:,2],risk_seed[seed]==2)
  m.update({'model':f'seed{seed}','hard_to_easy_count':int(((pd[seed]==0)&(y==1)).sum()),'hard_to_easy_rate':float(((pd[seed]==0)&(y==1)).sum()/y.sum()),'hard_class_recall':float(((pd[seed]==2)&(y==1)).sum()/y.sum())})
  table.append(m);curves[f'seed{seed}']=c
 mean_prob=probs.mean(0)
 mean_sev_probs=np.stack([arr(f'p_sev_{k}') for k in range(3)],axis=-1).mean(0)
 meanrisk=risk(mean_sev_probs.argmax(1),mean_prob.argmax(1))
 m,c=binary_metrics(y,mean_prob[:,2],meanrisk==2);m['model']='mean_probability_ensemble_additional';table.append(m);curves[m['model']]=c
 vote=flag_metrics(y,mr==2)
 vote.update({'model':'original_majority_vote','n':len(frames),'positives':int(y.sum()),'prevalence':float(y.mean()),'hard_to_easy_count':int(((md==0)&(y==1)).sum()),'hard_to_easy_rate':float(((md==0)&(y==1)).sum()/y.sum()),'hard_class_recall':float(((md==2)&(y==1)).sum()/y.sum())})
 table.append(vote)
 groups=[]
 for r,name in enumerate(['low','medium','high']):
  mask=mr==r
  groups.append({'risk':name,'n':int(mask.sum()),'true_q_mean':float(q[mask].mean()),'true_recall_mean':float(rec[mask].mean()),'true_hard_fraction':float(y[mask].mean())})
 csvout(root/'runs/A2/alert_metrics.csv',table)
 csvout(root/'runs/A2/original_majority_groups.csv',groups)
 # Coverage cut points are descriptive. No test-set threshold optimization.
 cover=[]
 for name,score in [(f'seed{s}',probs[s,:,2]) for s in range(3)]+[('mean_probability_ensemble_additional',mean_prob[:,2])]:
  for target in [.2,.4,.6,.8,1.]:
   threshold=float(np.sort(score)[max(0,int(np.ceil(target*len(score)))-1)])
   accept=score<=threshold
   cover.append({'model':name,'requested_coverage':target,'actual_coverage':float(accept.mean()),'maximum_accepted_p_hard':threshold,'accepted_hard_fraction':float(y[accept].mean()),'accepted_detector_loss_1_minus_q':float((1-q[accept]).mean())})
 csvout(root/'runs/A2/risk_coverage.csv',cover)
 metric_keys=['auroc','auprc_average_precision','binary_brier','binary_ece_15','alert_precision','alert_recall','alert_fpr','alert_fraction','hard_missed_by_alert_rate','hard_to_easy_rate','hard_class_recall']
 metric_summary={k:{'mean':float(np.mean([r[k] for r in table[:3]])),'std_ddof0':float(np.std([r[k] for r in table[:3]],ddof=0))} for k in metric_keys}
 # Resample the same frame indices jointly across all seeds to retain pairing.
 rng=np.random.default_rng(20261028)
 bootkeys=['auroc','auprc_average_precision','alert_precision','alert_recall','alert_fpr']
 boot={k:[] for k in bootkeys};bootvote={k:[] for k in ['alert_precision','alert_recall','alert_fpr']}
 for b in range(bootstrap):
  ix=rng.integers(0,len(frames),len(frames));yb=y[ix]
  if yb.min()==yb.max():continue
  mm=[binary_metrics(yb,probs[s,ix,2],risk_seed[s,ix]==2)[0] for s in range(3)]
  for k in bootkeys:boot[k].append(float(np.mean([x[k] for x in mm])))
  vm=flag_metrics(yb,mr[ix]==2)
  for k in bootvote:bootvote[k].append(vm[k])
  if b%500==0:print('BOOTSTRAP',b,'/',bootstrap,flush=True)
 cis={k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()}
 vote_cis={k:np.quantile(v,[.025,.975]).tolist() for k,v in bootvote.items()}
 result={'input_prediction_sha256':sha(source),'script_sha256':sha(Path(__file__)),'event':'true detectability == hard (class 2)','continuous_score':'softmax probability of hard, per seed; optional separate mean-probability ensemble','original_alert':'risk=detect_argmax+0.5*severity_argmax; low<=0.5, medium<=1.75, otherwise high; task-wise majority vote ties go to lowest class','thresholds_fitted_this_run':False,'calibration':'raw probabilities; binary Brier and 15 equal-width-bin ECE for hard event; no temperature fitting','pr_summary':'noninterpolated average precision, not trapezoidal PR-AUC','n_frames':len(frames),'n_hard':int(y.sum()),'per_model':table,'single_model_mean_std':metric_summary,'original_majority_groups':groups,'bootstrap':{'replicates':bootstrap,'seed':20261028,'unit':'frame; same draw shared by all seeds','single_model_mean_ci95':cis,'original_majority_ci95':vote_cis,'limitations':'Conditional on three trained models. No verified source-scene IDs; frame intervals do not establish scene independence or cover between-training-run uncertainty.'},'scope':'offline classification and association only, not collision risk or intervention safety'}
 jsonout(root/'runs/A2/summary.json',result);jsonout(root/'runs/A2/curves.json',curves)
 print('METRICS_DONE',json.dumps({'A1':summary,'A2_mean':metric_summary,'majority':vote,'groups':groups}),flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();evaluate(a.root,a.bootstrap)
