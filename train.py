"""Run one experiment from a JSON configuration, or print its command with --dry-run."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import argparse,json,sys,subprocess,hashlib,time
from pathlib import Path
ROOT=Path.cwd()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True,type=Path)
    ap.add_argument('--out-dir',type=Path,help='Required for a manual repeat; must be a new directory below the repository root')
    ap.add_argument('--seed',type=int)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args();cfg=json.loads(args.config.read_text())
    if args.seed is not None:cfg['seed']=args.seed
    target=(args.out_dir or Path(cfg['output_dir'])).resolve()
    assert ROOT in target.parents,'Outputs must remain below the repository root'
    cmd=[sys.executable,'-u',cfg['training_script']]+cfg['arguments']
    def replace(flag,value):
        if flag in cmd:cmd[cmd.index(flag)+1]=str(value)
        else:cmd.extend([flag,str(value)])
    replace(cfg['seed_flag'],cfg['seed']);replace(cfg['output_flag'],target)
    print(json.dumps({'configuration':cfg,'resolved_command':cmd,'output_dir':str(target)},indent=2),flush=True)
    if args.dry_run:return
    assert not target.exists(),'Use --out-dir with a NEW directory to avoid overwriting queued experiments'
    import torch
    assert torch.cuda.is_available(),'Refusing unnoticed CPU fallback'
    target.mkdir(parents=True)
    if cfg['task']=='text':
        import shutil
        for p in Path(cfg['label_dir']).glob('*labels.json'):shutil.copy2(p,target/p.name)
    manifest={'configuration':cfg,'resolved_command':cmd,'created':time.time(),'script_sha256':hashlib.sha256(Path(cfg['training_script']).read_bytes()).hexdigest(),'torch':torch.__version__,'device':torch.cuda.get_device_name(0)}
    (target/'manual_training_manifest.json').write_text(json.dumps(manifest,indent=2))
    env=dict(os.environ,PADU_RUN_OUT=str(target),PYTHONPATH=str(ROOT/'core'))
    with (target/'training.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        result=subprocess.CompletedProcess(cmd,process.wait())
    (target/'manual_exit.json').write_text(json.dumps({'returncode':result.returncode,'ended':time.time()}))
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
