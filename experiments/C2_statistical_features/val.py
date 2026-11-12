import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import argparse,json,csv,hashlib
import numpy as np,torch
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,f1_score,brier_score_loss
from model import SF
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--device',default='cpu',choices=['cpu','cuda']);a=ap.parse_args()
    assert Path.cwd() in a.output_dir.resolve().parents and not a.output_dir.exists()
    a.output_dir.mkdir(parents=True);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=ck['config']
    data=np.load(Path(c['feature_dir'])/'test.npz');x=(data['x']-ck['feature_mean'])/ck['feature_std'];y=data['y']
    model=SF(ck['input_dimension'],c['hidden_sizes'],c['dropout']).to(a.device).eval();model.load_state_dict(ck['model_state'],strict=True)
    with torch.inference_mode():p=model(torch.from_numpy(x).float().to(a.device)).sigmoid().cpu().numpy()
    pred=p>=c['test_threshold'];tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    result={'auroc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'f1':float(f1_score(y,pred)),'recall':float(tp/(tp+fn)),'fpr':float(fp/(tn+fp)),'n':len(y),'n_parameters':ck['n_parameters'],'config':c,'selection_epoch':ck['epoch'],'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()}
    (a.output_dir/'summary.json').write_text(json.dumps(result,indent=2))
    with (a.output_dir/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['frame_id','true_hard','p_hard','pred_hard'])
        w.writerows(zip(data['filenames'].tolist(),y.tolist(),p.tolist(),pred.astype(int).tolist()))
    (a.output_dir/'SUCCESS.json').write_text(json.dumps({'validated':True}))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
