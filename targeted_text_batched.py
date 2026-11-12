"""Generate targeted descriptions in batches and encode each description separately."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import json,time,argparse,hashlib
import numpy as np,torch

def load_model(model_path,max_pixels):
    from transformers import AutoModelForImageTextToText,AutoProcessor
    model=AutoModelForImageTextToText.from_pretrained(model_path,dtype=torch.float16,device_map='cuda').eval()
    processor=AutoProcessor.from_pretrained(model_path,max_pixels=max_pixels)
    processor.tokenizer.padding_side='left'
    model.generation_config.pad_token_id=processor.tokenizer.eos_token_id
    return model,processor

def generate(model,processor,paths,prompt,max_new_tokens,strip):
    import cv2
    from PIL import Image
    conversations=[]
    for path in paths:
        bgr=cv2.imread(str(path));assert bgr is not None,'Cannot decode '+str(path)
        image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        conversations.append([{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}])
    content=conversations if len(paths)>1 else conversations[0]
    kwargs=dict(add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)
    if len(paths)>1:kwargs['padding']=True
    inputs=processor.apply_chat_template(content,**kwargs).to(model.device)
    with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
    offset=inputs['input_ids'].shape[1]
    return [strip(processor.decode(row[offset:],skip_special_tokens=True).strip()) for row in out]

def encode_single(model,processor,description,layer):
    messages=[{'role':'user','content':[{'type':'text','text':description}]}]
    inputs=processor.apply_chat_template(messages,add_generation_prompt=False,tokenize=True,return_dict=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():out=model(**inputs,output_hidden_states=True)
    vector=out.hidden_states[layer][0].mean(0).to(torch.float16).cpu().numpy()
    assert np.isfinite(vector).all(),'Nonfinite text embedding'
    return vector

def generate_fallback(model,processor,paths,prompt,budget,strip):
    retry=False
    try:descriptions=generate(model,processor,paths,prompt,budget,strip)
    except torch.cuda.OutOfMemoryError:
        if len(paths)==1:raise
        retry=True
    if retry:
        torch.cuda.empty_cache();middle=max(1,len(paths)//2)
        return generate_fallback(model,processor,paths[:middle],prompt,budget,strip)+generate_fallback(model,processor,paths[middle:],prompt,budget,strip)
    return [(path,text,len(paths)) for path,text in zip(paths,descriptions)]

def run_from_module(g):
    ap=argparse.ArgumentParser()
    ap.add_argument('--xwod-root',default='data/xwod')
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--split',choices=['train','val','test'],default='test')
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--save-text',action='store_true',default=True)
    args=ap.parse_args()
    root=Path(g['__file__']).resolve().parents[1]
    plan_path=root/'C1_batch_plan.json'
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest()==g['BATCH_PLAN_SHA256'],'Pilot plan changed after generation script was prepared'
    plan=json.loads(plan_path.read_text())
    batch=plan['generation_batch_size'];assert plan['validated_text_equivalence'] and batch==g['GENERATION_BATCH_SIZE']
    assert plan['model_path']==g['MODEL_PATH'] and plan['prompt']==g['GEN_PROMPT'] and plan['max_pixels']==g['MAX_PIXELS'] and plan['max_new_tokens']==g['MAX_NEW_TOKENS']
    directory=Path(args.xwod_root)/('valid' if args.split=='val' else args.split)/'images'
    files=sorted(p.name for p in directory.glob('*.jpg'))
    if args.limit:files=files[:args.limit]
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    emb,valid,texts,progress=g['_resume_initialize'](args,files)
    indices={f:i for i,f in enumerate(files)};remaining=[directory/f for f in files if not valid[indices[f]]]
    model,processor=load_model(g['MODEL_PATH'],g['MAX_PIXELS'])
    suffix='_limit'+str(args.limit) if args.limit else ''
    protocol={'generation_batch_size_max':batch,'encoding_batch_size':1,'dtype':'float16','max_new_tokens':g['MAX_NEW_TOKENS'],'prompt':g['GEN_PROMPT'],'model_path':g['MODEL_PATH'],'left_padding':True,'OOM_policy':'retry smaller groups down to one; no invalid frames silently accepted','equivalence_pilot':plan,'throughput_timing_not_latency_benchmark':True}
    (out/(args.split+'_generation_protocol'+suffix+'.json')).write_text(json.dumps(protocol,indent=2))
    started=time.time();done_at_start=int(valid.sum());last_saved=done_at_start
    stats_path=out/(args.split+'_generation_batches'+suffix+'.jsonl')
    try:
        with stats_path.open('a') as stats:
            for start in range(0,len(remaining),batch):
                group=remaining[start:start+batch];t=time.time()
                results=generate_fallback(model,processor,group,g['GEN_PROMPT'],g['MAX_NEW_TOKENS'],g['_strip_thinking'])
                for path,text,actual_batch in results:
                    vector=encode_single(model,processor,text,g['LAYER_INDEX']);idx=indices[path.name]
                    emb[idx]=vector;texts[path.name]=text;valid[idx]=True
                    stats.write(json.dumps({'frame_id':path.name,'generation_batch_size':actual_batch})+'\n')
                stats.flush();done=int(valid.sum())
                if done-last_saved>=50 or start+batch>=len(remaining):
                    g['_save_generation_progress'](emb,valid,texts,files,progress);last_saved=done
                elapsed=time.time()-started;rate=(done-done_at_start)/elapsed
                print(json.dumps({'split':args.split,'complete':done,'total':len(files),'frames_per_second':rate,'estimated_remaining_seconds':(len(files)-done)/rate,'last_batch_seconds':time.time()-t}),flush=True)
    except BaseException:
        g['_save_generation_progress'](emb,valid,texts,files,progress)
        raise
    assert valid.all() and len(texts)==len(files)
    np.save(out/(args.split+'_text_emb'+suffix+'.npy'),emb)
    np.savez(out/(args.split+'_text_meta'+suffix+'.npz'),filenames=np.array(files),weathers=np.array([g['infer_weather'](f) for f in files]),valid=valid)
    (out/(args.split+'_descriptions'+suffix+'.json')).write_text(json.dumps(texts,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TARGETED_GENERATION_COMPLETE',args.split,len(files),flush=True)
