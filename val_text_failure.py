"""Evaluate targeted-text failure probabilities and risk flags on XWOD."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,time
import numpy as np,torch
from sklearn.metrics import average_precision_score,balanced_accuracy_score,brier_score_loss,confusion_matrix,roc_auc_score
O=Path.cwd()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def calculate(y,prob,pred,warning):
    hard=y==2;p=prob[:,2];tn,fp,fn,tp=confusion_matrix(hard,warning,labels=[False,True]).ravel();ece=0
    for k in range(15):
        keep=(p>=k/15)&(p<(k+1)/15 if k<14 else p<=1)
        if keep.any():ece+=keep.mean()*abs(p[keep].mean()-hard[keep].mean())
    return {'auroc':float(roc_auc_score(hard,p)),'ap':float(average_precision_score(hard,p)),'binary_brier':float(brier_score_loss(hard,p)),'ECE_15_equal_width':float(ece),'detect_balanced_accuracy':float(balanced_accuracy_score(y,pred)),'n':len(y),'n_hard':int(hard.sum()),'original_high_risk_warning':{'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn),'recall':float(tp/(tp+fn)),'FPR':float(fp/(fp+tn))}}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seed',type=int,choices=[0,1,2]);ap.add_argument('--batch',type=int,default=2);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    spec=importlib.util.spec_from_file_location('text_validation_original',O/'core/train_text_baseline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    labels=O/'data/understanding_labels';emb=O/'runs/C1/targeted_embeddings';dataset=m.TextEmbDataset('test',str(emb),str(labels));assert len(dataset)==2320
    with (O/'runs/A1_A2/frame_predictions.csv').open() as f:reference={r['frame_id']:r for r in csv.DictReader(f) if int(r['seed'])==0}
    assert {str(f) for f in dataset.items}==set(reference)
    root=O/'runs/C1/failure_validation';root.mkdir(exist_ok=True)
    seeds=[a.seed] if a.seed is not None else list(range(3));reports=[]
    for seed in seeds:
        target=root/f'seed{seed}'
        if (target/'SUCCESS.json').exists():reports.append(json.loads((target/'summary.json').read_text()));continue
        assert not target.exists(),'Partial independent validation requires inspection'
        target.mkdir();path=O/f'runs/C1/manual_repeat_seed{seed}/targeted_seed{seed}.pt';ck=torch.load(path,map_location='cpu',weights_only=False);state=ck['model_state']
        model=m.TextHeadModel(d_in=state['trunk.0.weight'].shape[1],d=state['trunk.0.weight'].shape[0]).float().cpu().eval();model.load_state_dict(state,strict=True)
        loader=torch.utils.data.DataLoader(dataset,batch_size=a.batch,shuffle=False,collate_fn=m.collate,num_workers=0)
        rows=[];ys=[];ps=[];predictions=[];warnings=[];offset=0
        with torch.inference_mode(),(target/'predictions.csv').open('w',newline='') as dst:
            writer=csv.DictWriter(dst,fieldnames=['frame_id','true_detect','pred_detect','pred_severity','p_det_0','p_det_1','p_det_2','risk']);writer.writeheader()
            for b in loader:
                output=model(b['x']);prob=output['detect'].softmax(-1).numpy();pred=prob.argmax(1);sev=output['severity'].argmax(1).numpy();y=b['detect'].numpy()
                assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1)
                for k in range(len(y)):
                    name=str(dataset.items[offset]);assert int(y[k])==int(reference[name]['true_detect'])
                    risk=pred[k]+.5*sev[k];level=0 if risk<=.5 else (1 if risk<=1.75 else 2)
                    row={'frame_id':name,'true_detect':int(y[k]),'pred_detect':int(pred[k]),'pred_severity':int(sev[k]),'p_det_0':float(prob[k,0]),'p_det_1':float(prob[k,1]),'p_det_2':float(prob[k,2]),'risk':level}
                    writer.writerow(row);ys.append(int(y[k]));ps.append(prob[k]);predictions.append(int(pred[k]));warnings.append(level==2);offset+=1
        assert offset==2320
        report={'seed':seed,'device':'cpu','dtype':'float32','batch_size':a.batch,'threads':1,'checkpoint':str(path),'checkpoint_sha256':sha(path),'checkpoint_parameters':ck['args'],'validation_selection':ck.get('validation_selection'),'n_parameters':sum(p.numel() for p in model.parameters()),'metrics':calculate(np.array(ys),np.array(ps,dtype=np.float64),np.array(predictions),np.array(warnings)),'predictions_sha256':sha(target/'predictions.csv'),'event_definition':'unchanged original A-hard, true_detect==2','no_test_fit':True,'note':'Independent CPU failure scoring; retain GPU four-task training evaluation separately.'}
        (target/'summary.json').write_text(json.dumps(report,indent=2));(target/'SUCCESS.json').write_text(json.dumps({'completed':time.time()}));reports.append(report)
    if a.seed is None:
        assert len(reports)==3
        aggregate={key:{'mean':float(np.mean([r['metrics'][key] for r in reports])),'sd_population':float(np.std([r['metrics'][key] for r in reports],ddof=0))} for key in ['auroc','ap','binary_brier','ECE_15_equal_width','detect_balanced_accuracy']}
        (root/'summary.json').write_text(json.dumps({'three_seed_aggregate':aggregate,'per_seed':reports},indent=2));(root/'SUCCESS.json').write_text(json.dumps({'completed':time.time()}))
    print('TEXT_FAILURE_VALIDATION_COMPLETE',seeds,flush=True)
if __name__=='__main__':main()
