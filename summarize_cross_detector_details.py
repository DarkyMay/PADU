"""Compute cross-detector event metrics using training-set thresholds."""
from pathlib import Path
import csv,json
import numpy as np
from sklearn.metrics import average_precision_score,balanced_accuracy_score,roc_auc_score,brier_score_loss
from scipy.stats import spearmanr
O=Path.cwd()

def metrics(rows,scores,thresholds):
    q=np.array([scores[r['frame_id']] for r in rows]);y=q<=thresholds['q1']
    levels=np.where(y,2,np.where(q<=thresholds['q2'],1,0));p=np.array([float(r['p_det_2']) for r in rows]);pred=np.array([int(r['pred_detect']) for r in rows]);warning=np.array([int(r['risk'])==2 for r in rows])
    tp=int((warning&y).sum());fp=int((warning&~y).sum());fn=int((~warning&y).sum());tn=int((~warning&~y).sum())
    def rate(a,b):return a/b if b else None
    ece=0
    for i in range(15):
        keep=(p>=i/15)&(p<(i+1)/15 if i<14 else p<=1)
        if keep.any():ece+=keep.mean()*abs(p[keep].mean()-y[keep].mean())
    return {'n':len(rows),'n_hard':int(y.sum()),'balanced_accuracy':float(balanced_accuracy_score(levels,pred)),'auroc':float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else None,'ap':float(average_precision_score(y,p)) if y.any() else None,'spearman_p_hard_vs_negative_q':float(spearmanr(p,-q).correlation),'binary_brier_raw':float(brier_score_loss(y,p)),'ECE_15_equal_width_raw':float(ece),'hard_to_easy_numerator':int((y&(pred==0)).sum()),'hard_to_easy_rate':rate(int((y&(pred==0)).sum()),int(y.sum())),'original_high_risk_warning':{'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':rate(tp,tp+fp),'recall':rate(tp,tp+fn),'FPR':rate(fp,fp+tn)}}

def main():
    out=O/'runs/C3/transfer_additional_metrics.json';assert not out.exists()
    with (O/'runs/A1_A2/frame_predictions.csv').open() as f:predictions=list(csv.DictReader(f))
    ref=json.loads((O/'runs/C3/rtdetrv2_r50_coco/summary.json').read_text())
    report={'threshold_fit':'Original A train thresholds are primary. B-specific tertiles use B training scores only.','probability_semantics':'Unchanged PaDU A-hard probability evaluated on each explicitly defined target; no calibration fit on validation/test.','risk_warning':'Original A-supervised predicted difficulty/severity and fixed risk rule; no B-supervised refit.','limitations':['Original data splits remain unchanged.','Three PaDU seeds do not measure detector-training variance.','Per-weather small/single-class event strata can have undefined AUROC.'],'arms':{}}
    for arm in ['A_to_A','COCO_pretrained_R50','XWOD_finetuned_R50']:
        if arm=='A_to_A':
            scores={r['frame_id']:float(r['true_q']) for r in predictions if int(r['seed'])==0};thresholds={'original_A_train':ref['thresholds']['original_A_train']}
        elif arm=='COCO_pretrained_R50':
            folder=O/'runs/C3/rtdetrv2_r50_coco'
            scores={r['frame_id']:r['q'] for r in map(json.loads,(folder/'test_frames.jsonl').read_text().splitlines()) if r['q'] is not None};thresholds=ref['thresholds']
        else:
            folder=O/'runs/C3/rtdetrv2_r50_finetuned_seed0'
            scores={r['frame_id']:r['q'] for r in map(json.loads,(folder/'eval_test/frames.jsonl').read_text().splitlines()) if r['q'] is not None}
            train=[r['q'] for r in map(json.loads,(folder/'eval_train/frames.jsonl').read_text().splitlines()) if r['q'] is not None]
            q1,q2=np.quantile(train,[1/3,2/3]).tolist();thresholds={'original_A_train':ref['thresholds']['original_A_train'],'B_train_tertiles':{'q1':q1,'q2':q2}}
        entry={'thresholds':thresholds,'definitions':{}}
        for definition,thr in thresholds.items():
            seeds=[]
            for seed in range(3):
                rows=sorted([r for r in predictions if int(r['seed'])==seed and r['frame_id'] in scores],key=lambda r:r['frame_id']);assert len(rows)==2320
                seeds.append({'seed':seed,'overall':metrics(rows,scores,thr),'by_weather':{weather:metrics([r for r in rows if r['weather']==weather],scores,thr) for weather in sorted({r['weather'] for r in rows})}})
            order=sorted([r['frame_id'] for r in predictions if int(r['seed'])==0]);by_seed={s:{r['frame_id']:r for r in predictions if int(r['seed'])==s} for s in range(3)}
            vote=[]
            for name in order:
                row=dict(by_seed[0][name])
                for key in ['pred_detect','pred_severity']:
                    votes=np.bincount([int(by_seed[s][name][key]) for s in range(3)],minlength=3);row[key]=int(votes.argmax())
                risk=row['pred_detect']+.5*row['pred_severity'];row['risk']=0 if risk<=.5 else (1 if risk<=1.75 else 2)
                # Probability metrics for this explicitly separate aggregation
                # are averaged probabilities, not a probability of hard voting.
                row['p_det_2']=float(np.mean([float(by_seed[s][name]['p_det_2']) for s in range(3)]));vote.append(row)
            voted=metrics(vote,scores,thr)
            aggregate={}
            for key in seeds[0]['overall']:
                values=[r['overall'][key] for r in seeds]
                if all(isinstance(x,(int,float)) and x is not None for x in values):aggregate[key]={'mean':float(np.mean(values)),'sd_population':float(np.std(values,ddof=0))}
            entry['definitions'][definition]={'per_seed':seeds,'three_seed_aggregate':aggregate,'original_task_vote':{k:v for k,v in voted.items() if k in ['n','n_hard','balanced_accuracy','hard_to_easy_numerator','hard_to_easy_rate','original_high_risk_warning']},'mean_probability_metrics_separate':{k:v for k,v in voted.items() if k in ['auroc','ap','binary_brier_raw','ECE_15_equal_width_raw','spearman_p_hard_vs_negative_q']}}
        report['arms'][arm]=entry
    out.write_text(json.dumps(report,indent=2));print('CROSS_DETECTOR_DETAILS_WRITTEN',out,flush=True)
if __name__=='__main__':main()
