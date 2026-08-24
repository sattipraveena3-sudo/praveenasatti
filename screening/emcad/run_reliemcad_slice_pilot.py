#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from scipy.ndimage import zoom

CLASSES={1:"spleen",2:"right kidney",3:"left kidney",4:"gallbladder",5:"pancreas",6:"liver",7:"stomach",8:"aorta"}

def remap(x):
    x=x.copy()
    for k in [5,9,10,12,13]: x[x==k]=0
    x[x==11]=5
    return x

def gamma_shift(sl,gamma):
    sl=sl.astype(np.float32,copy=False); lo=float(sl.min()); hi=float(sl.max())
    if hi-lo<1e-8:return sl.copy()
    x=np.clip((sl-lo)/(hi-lo),0,1); return (lo+np.power(x,gamma)*(hi-lo)).astype(np.float32)

def dice(a,b):
    a=a.astype(bool); b=b.astype(bool); d=a.sum()+b.sum()
    return 1.0 if d==0 else float(2*np.logical_and(a,b).sum()/d)

def iou(a,b):
    a=a.astype(bool); b=b.astype(bool); u=np.logical_or(a,b).sum()
    return 1.0 if u==0 else float(np.logical_and(a,b).sum()/u)

def ck_score(p):
    s=str(p).lower(); score=0
    score+=12 if p.name.lower()=="best.pth" else 0
    score+=8 if "pvt_v2_b2" in s or "pvtv2_b2" in s else 0
    score+=6 if "emcad" in s else 0
    score+=5 if "synapse" in s else 0
    return score,-len(s)

def clean_state(obj):
    if isinstance(obj,dict):
        for k in ["state_dict","model_state_dict","model","net"]:
            if k in obj and isinstance(obj[k],dict):obj=obj[k];break
    return {(k[7:] if k.startswith("module.") else k):v for k,v in obj.items()}

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--emcad-root',type=Path,required=True); ap.add_argument('--data-search',type=Path,required=True); ap.add_argument('--weights-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--gamma',type=float,default=1.8)
    a=ap.parse_args(); root=a.emcad_root.resolve(); sys.path.insert(0,str(root))
    from lib.networks import EMCADNet
    cases=[x.strip() for x in (root/'lists/lists_Synapse/test_vol.txt').read_text().splitlines() if x.strip()]; case=cases[0]
    dfs=list(a.data_search.resolve().rglob(case+'.npy.h5')); assert dfs,'data file missing'; data_file=dfs[0]
    cks=list(a.weights_root.resolve().rglob('*.pth'))+list(a.weights_root.resolve().rglob('*.pt')); assert cks,'checkpoint missing'; ck=sorted(cks,key=ck_score,reverse=True)[0]
    with h5py.File(data_file,'r') as f:image=f['image'][:]; label=remap(f['label'][:])
    fg=np.sum(label>0,axis=(1,2)); idx=int(np.argmax(fg)); sl=image[idx].astype(np.float32); gt=label[idx]
    present=[int(x) for x in np.unique(gt) if 1<=int(x)<=8]
    x,y=sl.shape; shifted=gamma_shift(sl,a.gamma)
    clean_r=zoom(sl,(224/x,224/y),order=3) if (x,y)!=(224,224) else sl
    shift_r=zoom(shifted,(224/x,224/y),order=3) if (x,y)!=(224,224) else shifted
    torch.set_num_threads(max(1,min(os.cpu_count() or 2,8))); device=torch.device('cpu')
    model=EMCADNet(num_classes=9,kernel_sizes=[1,3,5],expansion_factor=2,dw_parallel=True,add=True,lgag_ks=3,activation='relu6',encoder='pvt_v2_b2',pretrain=False,pretrained_dir='').to(device)
    inc=model.load_state_dict(clean_state(torch.load(ck,map_location=device)),strict=False); model.eval(); t0=time.time()
    inp=torch.from_numpy(np.stack([clean_r,shift_r])).unsqueeze(1).float().to(device)
    with torch.inference_mode():
        probs=torch.softmax(model(inp)[-1],dim=1); pp=torch.argmax(probs,dim=1).cpu().numpy(); ent=(-(probs*torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean(dim=(1,2))).cpu().numpy()
    pc,ps=pp[0],pp[1]
    if (x,y)!=(224,224):pc=zoom(pc,(x/224,y/224),order=0); ps=zoom(ps,(x/224,y/224),order=0)
    rows=[]
    for k in present:
        rows.append({'class_id':k,'class_name':CLASSES[k],'clean_dice':dice(pc==k,gt==k),'shift_dice':dice(ps==k,gt==k),'clean_iou':iou(pc==k,gt==k),'shift_iou':iou(ps==k,gt==k),'prediction_stability_dice':dice(pc==k,ps==k)})
    clean_mean=float(np.mean([r['clean_dice'] for r in rows])) if rows else 0.0; shift_mean=float(np.mean([r['shift_dice'] for r in rows])) if rows else 0.0; stability=float(np.mean([r['prediction_stability_dice'] for r in rows])) if rows else 1.0
    res={'experiment':'ReliEMCAD single-slice pilot','scope':'one automatically selected high-foreground slice; preliminary evidence only','case':case,'slice_index':idx,'foreground_pixels':int(fg[idx]),'present_classes':[CLASSES[k] for k in present],'gamma':a.gamma,'checkpoint_sha256':sha256(ck),'clean_present_class_mean_dice':clean_mean,'shifted_present_class_mean_dice':shift_mean,'dice_delta_shift_minus_clean':shift_mean-clean_mean,'clean_predictive_entropy':float(ent[0]),'shifted_predictive_entropy':float(ent[1]),'entropy_delta_shift_minus_clean':float(ent[1]-ent[0]),'clean_vs_shift_present_class_stability_dice':stability,'per_class':rows,'elapsed_seconds':round(time.time()-t0,2),'guardrail':'Pilot slice only; not a volume-level or benchmark-level result and not evidence of adaptation gain.'}
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);(out/'slice_pilot.json').write_text(json.dumps(res,indent=2));print('SLICE_PILOT',json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()
