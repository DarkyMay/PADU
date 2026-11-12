import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse,json,time,hashlib
from pathlib import Path
import torch,numpy as np
from ultralytics import YOLO
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));a=ap.parse_args();c=json.loads(a.config.read_text())
    folder=Path(c['feature_dir']);assert Path.cwd() in folder.resolve().parents;folder.mkdir(parents=True,exist_ok=True)
    detector=YOLO(c['detector_weights']);layer=detector.model.model[c['backbone_module_index']]
    assert type(layer).__name__==c['backbone_module_type']
    captured=[]
    def hook(module,inputs,output):
        x=output.detach().float().flatten(2)
        captured.append(torch.cat([x.mean(2),x.amax(2),x.std(2,unbiased=False)],1).cpu().numpy())
    handle=layer.register_forward_hook(hook)
    manifest={'config':c,'detector_sha256':hashlib.sha256(Path(c['detector_weights']).read_bytes()).hexdigest(),'layer_type':type(layer).__name__,'torch':torch.__version__,'splits':{}}
    try:
        for split,part in [('train','train'),('val','valid'),('test','test')]:
            target=folder/(split+'.npz');assert not target.exists(),'Existing features require explicit inspection'
            labels=json.loads((Path(c['label_dir'])/(split+'_detectability_labels.json')).read_text())
            understanding=json.loads((Path(c['label_dir'])/(split+'_understanding_labels.json')).read_text())
            with np.load(Path(c['token_root'])/(split+'_meta.npz'),allow_pickle=True) as meta:
                files=[str(f) for f,v in zip(meta['filenames'],meta['valid']) if v and str(f) in labels and str(f) in understanding]
            feats=[];ys=[]
            for i,name in enumerate(files):
                path=Path('data/xwod')/part/'images'/name;captured.clear()
                detector.predict(str(path),imgsz=c['image_size'],device=0,conf=.25,verbose=False)
                assert captured and captured[-1].shape[0]==1
                x=captured[-1][0];assert np.isfinite(x).all()
                feats.append(x);ys.append(int(labels[name]['difficulty']==2))
                if (i+1)%100==0:print('SF_FEATURES',split,i+1,len(files),flush=True)
            np.savez(target,x=np.stack(feats),y=np.array(ys),filenames=np.array(files))
            manifest['splits'][split]={'n':len(files),'dimension':len(feats[0]),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    finally:handle.remove()
    (folder/'SUCCESS.json').write_text(json.dumps({'completed':time.time()}))
if __name__=='__main__':main()
