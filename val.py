"""Inspect a checkpoint or evaluate it with the matching experiment configuration."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,argparse,json,hashlib,importlib.util,time
from pathlib import Path
ROOT=Path.cwd()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True,type=Path)
    ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--inspect-only',action='store_true')
    ap.add_argument('--output',type=Path)
    a=ap.parse_args();cfg=json.loads(a.config.read_text())
    import torch
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    saved=ck.get('config',ck.get('args',{}))
    report={'checkpoint':str(a.checkpoint),'sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'checkpoint_parameters':saved,'experiment_configuration':cfg,'epoch':ck.get('epoch'),'best_round':ck.get('best_round'),'tag':ck.get('tag'),'torch':torch.__version__,'device':a.device,'time':time.time()}
    state=ck.get('best_state') or ck.get('model_state') or ck.get('returned_model_state')
    if state is not None:report['checkpoint_state_tensor_elements']=sum(t.numel() for t in state.values())
    if a.inspect_only:print(json.dumps(report,indent=2,default=str));return
    sys.path.insert(0,'core')
    if cfg['task']=='federation':
        f=module('revision_fed_validation',cfg['training_script'])
        factory=f.import_model('core')
        ds=f.Split('XWOD','test',cfg['token_root'],cfg['label_dir'],'memmap',torch.device(a.device))
        model=f.new_model(factory,torch.device(a.device),saved,seed=saved['seed'])
        tag=ck.get('tag','');is_private=tag in ['FedPer','local-only']
        if is_private:
            paths=list((a.checkpoint.parent.parent/'partitions').glob('*.json'))
            assert len(paths)==1,'Specify an unambiguous saved client partition'
            clients,va,te=json.loads(paths[0].read_text())['partition_positions']
            import numpy as np
            for c in clients:c['idx']=np.array(c['idx'],dtype=int)
            te={k:np.array(v,dtype=int) for k,v in te.items()}
            private=(ck.get('best_private') or ck['final_private'])
            base=ck.get('best_state') if tag=='FedPer' else None
            report['metrics']=f.personalized_test(factory,private,clients,ds,torch.device(a.device),saved,te,base_state=base)
        else:
            assert state is not None;model.load_state_dict(state,strict=True)
            import numpy as np
            report['metrics']=f.strip_metrics(f.evaluate(model,ds,np.arange(ds.n),torch.device(a.device),a.batch))
        report['n_parameters']=sum(p.numel() for p in model.parameters())
    elif cfg['task']=='layer':
        from unified_model import UnifiedWeatherModel,WeatherUnderstandingDataset,collate
        from torch.utils.data import DataLoader
        from eval_test import evaluate
        assert Path(cfg['token_root']).exists(),'Layer cache was temporary; regenerate using the archived extraction command and cache manifest before validating.'
        model=UnifiedWeatherModel(d_model=saved.get('d_model',384),n_layers=saved.get('n_layers',3),n_heads=saved.get('n_heads',6),dropout=saved.get('dropout',.1)).to(a.device)
        model.load_state_dict(state,strict=True)
        ds=WeatherUnderstandingDataset('test',cfg['token_root'],cfg['label_dir'])
        metrics,scm,dcm=evaluate(model,DataLoader(ds,batch_size=a.batch,collate_fn=collate),a.device)
        report.update(metrics=metrics,severity_confusion=scm.tolist(),detect_confusion=dcm.tolist(),n_parameters=sum(p.numel() for p in model.parameters()))
    elif cfg['task']=='text':
        from train_text_baseline import TextHeadModel,TextEmbDataset,collate,evaluate
        from torch.utils.data import DataLoader
        model=TextHeadModel().to(a.device);model.load_state_dict(state,strict=True)
        ds=TextEmbDataset('test',cfg['embedding_root'],cfg['label_dir'])
        report.update(metrics=evaluate(model,DataLoader(ds,batch_size=a.batch,collate_fn=collate),a.device),n_parameters=sum(p.numel() for p in model.parameters()))
    else:raise ValueError('Use the dedicated C3_rtdetrv2_r50/val.py for the detector itself')
    print(json.dumps(report,indent=2,default=str))
    if a.output:
        assert ROOT in a.output.resolve().parents and not a.output.exists()
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,default=str))
if __name__=='__main__':main()
