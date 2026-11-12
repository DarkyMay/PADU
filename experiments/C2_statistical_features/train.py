import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse,json,random,time,copy
from pathlib import Path
import numpy as np,torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import roc_auc_score
from model import SF
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args();c=json.loads(a.config.read_text());c['seed']=a.seed
    assert Path.cwd() in a.out_dir.resolve().parents and not a.out_dir.exists()
    a.out_dir.mkdir(parents=True);(a.out_dir/'config.json').write_text(json.dumps(c,indent=2))
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    tr=np.load(Path(c['feature_dir'])/'train.npz');va=np.load(Path(c['feature_dir'])/'val.npz')
    mean=tr['x'].mean(0);std=tr['x'].std(0).clip(1e-6)
    x=torch.from_numpy((tr['x']-mean)/std).float();y=torch.from_numpy(tr['y']).float()
    vx=torch.from_numpy((va['x']-mean)/std).float().cuda();vy=va['y']
    model=SF(x.shape[1],c['hidden_sizes'],c['dropout']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y)-y.sum())/y.sum()],device='cuda'))
    loader=DataLoader(TensorDataset(x,y),batch_size=c['batch_size'],shuffle=True)
    best=-1;bad=0
    with (a.out_dir/'history.jsonl').open('w') as history:
        for ep in range(c['epochs']):
            model.train()
            for bx,by in loader:
                loss=criterion(model(bx.cuda()),by.cuda());assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            model.eval()
            with torch.inference_mode():p=model(vx).sigmoid().cpu().numpy()
            value=float(roc_auc_score(vy,p));row={'epoch':ep+1,'validation_auroc':value}
            history.write(json.dumps(row)+'\n');history.flush();print(json.dumps(row),flush=True)
            if value>best+1e-4:
                best=value;bad=0
                torch.save({'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':c,'epoch':ep+1,'validation_auroc':value,'feature_mean':mean,'feature_std':std,'input_dimension':x.shape[1],'n_parameters':sum(p.numel() for p in model.parameters())},a.out_dir/'best.pt')
            else:bad+=1
            if bad>=c['patience']:break
    (a.out_dir/'SUCCESS.json').write_text(json.dumps({'completed':time.time(),'best_validation_auroc':best}))
if __name__=='__main__':main()
