"""Measure batch-one PaDU latency with a shared frozen VLM encoder."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time,csv,hashlib,random
from pathlib import Path
R=Path.cwd();D=R/'runs/A3';D.mkdir(parents=True,exist_ok=True)
def main():
    import numpy as np,torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModelForImageTextToText,AutoProcessor
    sys.path.insert(0,str(R/'core'))
    from unified_model import UnifiedWeatherModel
    assert torch.cuda.is_available()
    # Require an idle GPU for timing.
    import subprocess
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert all(p.strip()==str(os.getpid()) for p in active.splitlines()),'GPU is occupied: '+active
    modelpath=R/'models/Qwen3.5-9B'
    start=time.perf_counter()
    vlm=AutoModelForImageTextToText.from_pretrained(modelpath,dtype=torch.bfloat16,device_map='cuda').eval()
    processor=AutoProcessor.from_pretrained(modelpath,max_pixels=1280*1280)
    decoders=[];hashes={}
    for seed in range(3):
        path=R/f'runs/weatherfilm_runs/uwm_main_seed{seed}/best_uwm.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False);a=ck['args']
        m=UnifiedWeatherModel(d_model=a.get('d_model',384),n_layers=a.get('n_layers',3),n_heads=a.get('n_heads',6),dropout=a.get('dropout',.1)).float().cuda().eval()
        m.load_state_dict(ck['model_state'],strict=True);decoders.append(m)
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    torch.cuda.synchronize();load_seconds=time.perf_counter()-start
    pad=processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    names=sorted((R/'data/xwod/test/images').glob('*.jpg'));random.Random(20261030).shuffle(names)
    chosen=names[:210];(D/'frame_list.json').write_text(json.dumps([p.name for p in chosen]))
    cached=np.load(R/'data/weather_patch_tokens/test_tokens.npy',mmap_mode='r')
    with np.load(R/'data/weather_patch_tokens/test_meta.npz',allow_pickle=True) as meta:
        cache_rows={str(name):i for i,name in enumerate(meta['filenames'])}
    replay=[]
    def timed(action):
        torch.cuda.synchronize();t=time.perf_counter();v=action();torch.cuda.synchronize()
        return v,1000*(time.perf_counter()-t)
    def encode(inputs):
        output=vlm(**inputs,output_hidden_states=True,return_dict=True)
        mask=inputs['input_ids'][0]==pad
        t,h,w=[int(v) for v in inputs['image_grid_thw'][0].tolist()];h//=2;w//=2
        assert t==1 and h*w==int(mask.sum())
        grid=output.hidden_states[-4][0,mask].reshape(h,w,4096).permute(2,0,1).unsqueeze(0).float()
        return F.adaptive_avg_pool2d(grid,(12,16)).squeeze(0).permute(1,2,0).reshape(1,192,4096).to(torch.float16).float()
    fields=['frame_id','warmup','decode_ms','preprocess_ms','transfer_ms','encoder_pool_ms','single_decoder_ms','extra_two_decoders_ms','single_pipeline_ms','vote_pipeline_ms','image_tokens','peak_allocated_bytes','peak_reserved_bytes','text_generation_reencode_ms']
    rows=[]
    with torch.inference_mode(),(D/'latency_frames.csv').open('w',newline='') as dst:
        writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader()
        for i,path in enumerate(chosen):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();overall=time.perf_counter()
            image,decode=timed(lambda:Image.open(path).convert('RGB'))
            msg=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Describe the weather degradation in this driving scene.'}]}]
            inputs,prep=timed(lambda:processor.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt'))
            inputs,transfer=timed(lambda:inputs.to('cuda'))
            tokens,encoding=timed(lambda:encode(inputs))
            first,single=timed(lambda:decoders[0](tokens))
            single_wall=1000*(time.perf_counter()-overall)
            def vote():
                outputs=[first]+[m(tokens) for m in decoders[1:]]
                # Resolve voting ties by the lowest class index, then apply the risk rule.
                difficulty=torch.stack([o['detect'].argmax(1) for o in outputs]).mode(0).values
                severity=torch.stack([o['severity'].argmax(1) for o in outputs]).mode(0).values
                risk=difficulty+.5*severity
                return torch.where(risk<=.5,0,torch.where(risk<=1.75,1,2))
            _,extra=timed(vote);vote_wall=1000*(time.perf_counter()-overall)
            row=dict(zip(fields,[path.name,i<10,decode,prep,transfer,encoding,single,extra,single_wall,vote_wall,int((inputs['input_ids']==pad).sum()),torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved(),None]))
            if i<3:
                # Compare live and cached encodings during warmup, outside the timed segments.
                old=np.asarray(cached[cache_rows[path.name]],dtype=np.float32).reshape(-1)
                live=tokens[0].detach().cpu().numpy().reshape(-1)
                cosine=float(np.dot(old,live)/(np.linalg.norm(old)*np.linalg.norm(live)))
                replay.append({'frame_id':path.name,'cosine':cosine,'max_abs_difference':float(np.max(np.abs(old-live))),'mean_abs_difference':float(np.mean(np.abs(old-live))),'bitwise_equal':bool(np.array_equal(old,live))})
            # Time text generation on 40 frames with a 128-token budget.
            if 10<=i<50:
                textmsg=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Describe the weather conditions and the visible objects (vehicles, people) in this driving scene.'}]}]
                def text_arm():
                    x=processor.apply_chat_template(textmsg,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False).to('cuda')
                    g=vlm.generate(**x,max_new_tokens=128,do_sample=False,pad_token_id=processor.tokenizer.eos_token_id)
                    desc=processor.decode(g[0][x['input_ids'].shape[1]:],skip_special_tokens=True)
                    txt=processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':desc}]}],add_generation_prompt=False,tokenize=True,return_dict=True,return_tensors='pt').to('cuda')
                    return vlm(**txt,output_hidden_states=True).hidden_states[-4][0].mean(0)
                _,row['text_generation_reencode_ms']=timed(text_arm)
            writer.writerow(row);dst.flush();rows.append(row)
            del tokens,inputs,first,image
            if (i+1)%10==0:print('A3_FRAMES',i+1,len(chosen),flush=True)
    measured=[r for r in rows if not r['warmup']]
    summary={'settings':{'batch':1,'max_pixels':1280*1280,'layer':-4,'pool':[12,16],'vlm_dtype':'bfloat16','decoder_dtype':'float32','warmup':10,'n':200,'text_n':40,'text_max_new_tokens':128,'text_readout_included':False,'timing':'wall clock with CUDA synchronize per segment; filesystem cache not flushed; excludes object detector','shared_encoding':True,'load_seconds':load_seconds},'checkpoints':hashes,'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'metrics':{}}
    for key in fields[2:]:
        values=[r[key] for r in measured if r[key] is not None]
        if values:summary['metrics'][key]={'mean':float(np.mean(values)),'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'p99':float(np.quantile(values,.99)),'n':len(values)}
    summary['original_cache_replay_check']=replay
    (D/'summary.json').write_text(json.dumps(summary,indent=2))
    (D/'SUCCESS.json').write_text(json.dumps({'completed':time.time()}))
if __name__=='__main__':main()
