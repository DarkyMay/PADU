"""Measure complete targeted-text and statistical-feature baseline latency."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import ast,csv,hashlib,importlib.util,json,subprocess,sys,time
import numpy as np,torch
O=Path.cwd();R=O;D=O/'runs/A3_complete_baselines'
sys.path.insert(0,str(O))

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def timed(f):
    torch.cuda.synchronize();t=time.perf_counter();value=f();torch.cuda.synchronize()
    return value,1000*(time.perf_counter()-t)

def stats(values):
    return {'n':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'p99':float(np.quantile(values,.99))}

def main():
    assert torch.cuda.is_available()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert all(p.strip()==str(os.getpid()) for p in active.splitlines()),'GPU is occupied: '+active
    assert not D.exists(),'Partial benchmark requires inspection and a separate rerun directory'
    names=json.loads((O/'runs/A3/frame_list.json').read_text());assert len(names)==210
    for seed in range(3):assert (O/f'runs/C1/manual_repeat_seed{seed}/targeted_seed{seed}.pt').exists()
    D.mkdir();(D/'frame_list.json').write_text(json.dumps(names))
    info={'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'CUDA_runtime':torch.version.cuda,'NVIDIA_driver_version':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),'script_sha256':sha(Path(__file__)),'batch_size':1,'timing':'wall clock with CUDA synchronization, filesystem cache not flushed; model loading excluded and ten warmup frames discarded','object_detector_in_targeted_text':False,'targeted_text_dtype':'float16 VLM, float32 trained head','targeted_text_model_path':str(R/'models/Qwen3.5-9B'),'targeted_text_max_pixels':1280*1280,'targeted_text_encoding_layer':-4,'SF_dtype':'float32 detector and trained head','targeted_text_warmup':10,'targeted_text_measured':40,'SF_warmup':10,'SF_measured':200,'shared_A3_frame_list_sha256':sha(O/'runs/A3/frame_list.json')}
    (D/'manifest.json').write_text(json.dumps(info,indent=2))
    from targeted_text_batched import load_model,generate,encode_single
    original=module('text_training_timing',R/'core/train_text_baseline.py')
    protocol=module('text_generation_protocol_timing',R/'core/gen_text_embeddings.py')
    prompt=json.loads((O/'gpu_representation/targeted_protocol.json').read_text())['prompt']
    path=O/'runs/C1/manual_repeat_seed0/targeted_seed0.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    readout=original.TextHeadModel().float().cuda().eval();readout.load_state_dict(checkpoint['model_state'],strict=True)
    info['targeted_text_head_sha256']=sha(path);info['targeted_text_prompt']=prompt;info['targeted_text_max_new_tokens']=128
    info['targeted_text_head_parameters']=sum(p.numel() for p in readout.parameters())
    vlm,processor=load_model(str(R/'models/Qwen3.5-9B'),1280*1280)
    text_rows=[]
    with (D/'targeted_text_frames.jsonl').open('w') as dst,torch.inference_mode():
        for i,name in enumerate(names[:50]):
            torch.cuda.reset_peak_memory_stats()
            def pipeline():
                description=generate(vlm,processor,[R/'data/xwod/test/images'/name],prompt,128,protocol._strip_thinking)[0]
                vector=encode_single(vlm,processor,description,-4)
                result=readout(torch.from_numpy(vector.copy()).float().unsqueeze(0).cuda())
                _=result['detect'].argmax(1).cpu().tolist()
                return description
            text,elapsed=timed(pipeline)
            row={'frame_id':name,'warmup':i<10,'pipeline_ms':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'response':text}
            text_rows.append(row);dst.write(json.dumps(row)+'\n');dst.flush()
            print('COMPLETE_TARGETED_TIMING',i+1,50,flush=True)
    del vlm,processor,readout,checkpoint;import gc;gc.collect();torch.cuda.empty_cache()
    sf_module=module('sf_head_timing',O/'experiments/C2_statistical_features/model.py')
    path=O/'runs/C2/seed0/best.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False);config=checkpoint['config']
    head=sf_module.SF(checkpoint['input_dimension'],config['hidden_sizes'],config['dropout']).float().cuda().eval();head.load_state_dict(checkpoint['model_state'],strict=True)
    from ultralytics import YOLO
    detector=YOLO(config['detector_weights']);captured=[]
    layer=detector.model.model[config['backbone_module_index']];assert type(layer).__name__==config['backbone_module_type']
    hook_times=[]
    def hook(m,inputs,output):
        def features():
            x=output.detach().float().flatten(2)
            return torch.cat([x.mean(2),x.amax(2),x.std(2,unbiased=False)],1).cpu().numpy()
        vector,milliseconds=timed(features);captured.append(vector);hook_times.append(milliseconds)
    handle=layer.register_forward_hook(hook);sf_rows=[]
    info.update(SF_head_sha256=sha(path),SF_detector_sha256=sha(Path(config['detector_weights'])),SF_head_parameters=sum(p.numel() for p in head.parameters()),SF_image_size=config['image_size'],SF_detector_confidence=.25,SF_normalization='training mean/std, original CPU normalization and feature roundtrip included')
    try:
        with (D/'SF_frames.jsonl').open('w') as dst,torch.inference_mode():
            for i,name in enumerate(names):
                captured.clear();hook_times.clear();torch.cuda.reset_peak_memory_stats();segments={}
                def pipeline():
                    detector.predict(str(R/'data/xwod/test/images'/name),imgsz=config['image_size'],device=0,conf=.25,verbose=False)
                    assert captured and captured[-1].shape==(1,checkpoint['input_dimension'])
                    def monitoring_head():
                        x=(captured[-1]-checkpoint['feature_mean'])/checkpoint['feature_std']
                        return head(torch.from_numpy(x).float().cuda()).sigmoid().cpu().tolist()
                    value,segments['normalization_transfer_and_head_ms']=timed(monitoring_head)
                    return value
                _,elapsed=timed(pipeline)
                row={'frame_id':name,'warmup':i<10,'pipeline_ms':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'feature_pooling_and_CPU_copy_ms':hook_times[-1],**segments}
                sf_rows.append(row);dst.write(json.dumps(row)+'\n');dst.flush()
                if (i+1)%10==0:print('COMPLETE_SF_TIMING',i+1,len(names),flush=True)
    finally:handle.remove()
    detector_rows=[]
    with (D/'SF_detector_only_frames.jsonl').open('w') as dst,torch.inference_mode():
        for i,name in enumerate(names):
            def detector_only():return detector.predict(str(R/'data/xwod/test/images'/name),imgsz=config['image_size'],device=0,conf=.25,verbose=False)
            _,elapsed=timed(detector_only)
            row={'frame_id':name,'warmup':i<10,'detector_only_pipeline_ms':elapsed};detector_rows.append(row);dst.write(json.dumps(row)+'\n');dst.flush()
    result={'settings':info,'targeted_text_pipeline_ms':stats([r['pipeline_ms'] for r in text_rows if not r['warmup']]),'SF_detector_and_monitor_pipeline_ms':stats([r['pipeline_ms'] for r in sf_rows if not r['warmup']])}
    result['SF_detector_only_pipeline_ms']=stats([r['detector_only_pipeline_ms'] for r in detector_rows if not r['warmup']])
    for key in ['feature_pooling_and_CPU_copy_ms','normalization_transfer_and_head_ms']:
        result['SF_'+key]=stats([r[key] for r in sf_rows if not r['warmup']])
    result['SF_pooling_normalization_and_head_segments_sum_ms']=stats([r['feature_pooling_and_CPU_copy_ms']+r['normalization_transfer_and_head_ms'] for r in sf_rows if not r['warmup']])
    info['SF_segments']='CUDA synchronized segments; the full pipeline includes their synchronization overhead. Detector-only is a separate warm run, not subtracted as an exact marginal overhead.'
    with (O/'runs/A3/latency_frames.csv').open() as f:old={r['frame_id']:r for r in csv.DictReader(f)}
    paired=[r['frame_id'] for r in text_rows if not r['warmup']]
    result['PaDU_single_pipeline_same_40_frames_ms']=stats([float(old[n]['single_pipeline_ms']) for n in paired])
    result['targeted_text_over_PaDU_ratio_of_means_same_40_frames']=result['targeted_text_pipeline_ms']['mean']/result['PaDU_single_pipeline_same_40_frames_ms']['mean']
    result['limitations']=['PaDU and baseline times were measured in separate exclusive-GPU runs, not simultaneously.','PaDU uses the original bfloat16 encoder; targeted text retains the original float16 baseline protocol.','SF includes the detector; PaDU and targeted text exclude the supervised detector.','Offline generation throughput batching does not enter these batch-one latency measurements.']
    (D/'manifest.json').write_text(json.dumps(info,indent=2));(D/'summary.json').write_text(json.dumps(result,indent=2));(D/'SUCCESS.json').write_text(json.dumps({'completed':time.time()}))
    print('REMAINING_PIPELINE_TIMINGS_COMPLETE',flush=True)
if __name__=='__main__':main()
