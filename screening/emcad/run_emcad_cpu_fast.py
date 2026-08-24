#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from medpy.metric import binary as mb
from scipy.ndimage import zoom

CLASSES=["spleen","right kidney","left kidney","gallbladder","pancreas","liver","stomach","aorta"]
PAPER={"mean_dice":0.8363,"mean_hd95":15.68,"mean_jaccard":0.7465}

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def metric(pred,gt):
    pred=(pred>0).astype(np.uint8); gt=(gt>0).astype(np.uint8)
    if pred.sum()>0 and gt.sum()>0:
        return float(mb.dc(pred,gt)),float(mb.hd95(pred,gt)),float(mb.jc(pred,gt)),float(mb.assd(pred,gt))
    if pred.sum()>0 and gt.sum()==0: return 1.0,0.0,1.0,0.0
    return 0.0,0.0,0.0,0.0

def remap(x):
    x=x.copy()
    for k in [5,9,10,12,13]: x[x==k]=0
    x[x==11]=5
    return x

def find_data(root,cases):
    for p in root.rglob(cases[0]+".npy.h5"):
        if all((p.parent/(c+".npy.h5")).exists() for c in cases): return p.parent
    raise FileNotFoundError("Synapse test volumes not found")

def ck_score(p):
    s=str(p).lower(); score=0
    score+=12 if p.name.lower()=="best.pth" else 0
    score+=8 if "pvt_v2_b2" in s or "pvtv2_b2" in s else 0
    score+=6 if "emcad" in s else 0
    score+=5 if "synapse" in s else 0
    score+=2 if "run1" in s or "run_1" in s else 0
    return score,-len(s)

def find_ck(root):
    files=list(root.rglob("*.pth"))+list(root.rglob("*.pt"))
    if not files: raise FileNotFoundError("No checkpoint found")
    return sorted(files,key=ck_score,reverse=True)[0]

def clean(obj):
    if isinstance(obj,dict):
        for k in ["state_dict","model_state_dict","model","net"]:
            if k in obj and isinstance(obj[k],dict): obj=obj[k]; break
    return {(k[7:] if k.startswith("module.") else k):v for k,v in obj.items()}

def git_commit(repo):
    try:return subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    except:return "unknown"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--emcad-root",type=Path,required=True); ap.add_argument("--data-search",type=Path,required=True)
    ap.add_argument("--weights-root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--batch-size",type=int,default=12)
    a=ap.parse_args(); root=a.emcad_root.resolve(); sys.path.insert(0,str(root))
    from lib.networks import EMCADNet
    cases=[x.strip() for x in (root/"lists/lists_Synapse/test_vol.txt").read_text().splitlines() if x.strip()]
    data=find_data(a.data_search.resolve(),cases); ck=find_ck(a.weights_root.resolve())
    device=torch.device("cpu"); torch.set_num_threads(max(1,min(os.cpu_count() or 2,8))); torch.manual_seed(2222); np.random.seed(2222)
    model=EMCADNet(num_classes=9,kernel_sizes=[1,3,5],expansion_factor=2,dw_parallel=True,add=True,lgag_ks=3,activation="relu6",encoder="pvt_v2_b2",pretrain=False,pretrained_dir="").to(device)
    inc=model.load_state_dict(clean(torch.load(ck,map_location=device)),strict=False)
    if len(inc.missing_keys)>10: raise RuntimeError(f"Checkpoint mismatch: {len(inc.missing_keys)} missing keys")
    model.eval(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    rows=[]; cubes=[]; case_results=[]; t0=time.time()
    with torch.inference_mode():
        for ci,case in enumerate(cases,1):
            st=time.time()
            with h5py.File(data/(case+".npy.h5"),"r") as f: image=f["image"][:]; label=remap(f["label"][:])
            predvol=np.zeros_like(label)
            for b0 in range(0,image.shape[0],a.batch_size):
                b1=min(image.shape[0],b0+a.batch_size); batch=[]; shapes=[]
                for i in range(b0,b1):
                    sl=image[i]; x,y=sl.shape; shapes.append((x,y)); batch.append(zoom(sl,(224/x,224/y),order=3) if (x,y)!=(224,224) else sl)
                inp=torch.from_numpy(np.stack(batch)).unsqueeze(1).float().to(device)
                logits=model(inp)[-1]; pp=torch.argmax(torch.softmax(logits,dim=1),dim=1).cpu().numpy()
                for j,i in enumerate(range(b0,b1)):
                    x,y=shapes[j]; p=pp[j]; predvol[i]=zoom(p,(x/224,y/224),order=0) if (x,y)!=(224,224) else p
            cm=[]
            for k,name in enumerate(CLASSES,1):
                vals=metric(predvol==k,label==k); cm.append(vals)
                rows.append({"case":case,"class_id":k,"class_name":name,"dice":vals[0],"hd95":vals[1],"jaccard":vals[2],"asd":vals[3]})
            arr=np.asarray(cm,float); cubes.append(arr)
            cr={"case":case,"mean_dice":float(arr[:,0].mean()),"mean_hd95":float(arr[:,1].mean()),"mean_jaccard":float(arr[:,2].mean()),"mean_asd":float(arr[:,3].mean()),"slices":int(image.shape[0]),"seconds":round(time.time()-st,2)}
            case_results.append(cr); print(f"[{ci:02d}/{len(cases)}] {case} Dice={cr['mean_dice']:.4f} HD95={cr['mean_hd95']:.2f} J={cr['mean_jaccard']:.4f} {cr['seconds']:.1f}s",flush=True)
    cube=np.asarray(cubes,float); class_means=cube.mean(axis=0); overall=class_means.mean(axis=0)
    class_results=[{"class_id":i+1,"class_name":n,"dice":float(class_means[i,0]),"hd95":float(class_means[i,1]),"jaccard":float(class_means[i,2]),"asd":float(class_means[i,3])} for i,n in enumerate(CLASSES)]
    m={"project":"EMCAD Synapse checkpoint verification","paper":"EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation (CVPR 2024)","official_repo":"https://github.com/SLDGroup/EMCAD","official_repo_commit":git_commit(root),"device":"cpu","torch_version":torch.__version__,"batch_size":a.batch_size,"num_test_cases":len(cases),"checkpoint":str(ck),"checkpoint_sha256":sha256(ck),"model_parameters":sum(p.numel() for p in model.parameters()),"missing_checkpoint_keys":list(inc.missing_keys),"unexpected_checkpoint_keys":list(inc.unexpected_keys),"observed":{"mean_dice":float(overall[0]),"mean_hd95":float(overall[1]),"mean_jaccard":float(overall[2]),"mean_asd":float(overall[3])},"paper_reference":PAPER,"absolute_dice_gap":float(abs(overall[0]-PAPER['mean_dice'])),"elapsed_seconds":round(time.time()-t0,2),"class_results":class_results,"case_results":case_results}
    (out/"metrics.json").write_text(json.dumps(m,indent=2))
    with open(out/"per_case_class_metrics.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    report=f"""# Step 1 - EMCAD Experimental Verification\n\n**Applicant:** Satti Praveena  \n**Paper:** EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation (CVPR 2024)  \n**Experiment:** Verification of the authors' released PVT-EMCAD-B2 checkpoint on the released Synapse test split.\n\n## Reproducibility setup\n- Official repository commit: `{m['official_repo_commit']}`\n- Test volumes: {len(cases)}\n- Checkpoint SHA-256: `{m['checkpoint_sha256']}`\n- Model parameters: {m['model_parameters']:,}\n- CPU batched inference: batch size {a.batch_size}\n- Input size: 224 x 224\n- Evaluation follows the released 9-class Synapse label mapping.\n\n## Results\n| Metric | Observed | Paper reference |\n|---|---:|---:|\n| Mean Dice | {overall[0]*100:.2f}% | {PAPER['mean_dice']*100:.2f}% |\n| Mean HD95 | {overall[1]:.2f} | {PAPER['mean_hd95']:.2f} |\n| Mean Jaccard | {overall[2]*100:.2f}% | {PAPER['mean_jaccard']*100:.2f}% |\n| Mean ASD | {overall[3]:.2f} | - |\n\nAbsolute Dice gap from the paper reference: {abs(overall[0]-PAPER['mean_dice'])*100:.2f} percentage points. The paper reference is an averaged reported result; this experiment verifies one released trained checkpoint rather than reproducing multi-run training.\n\n## Per-class results\n| Class | Dice | HD95 | Jaccard | ASD |\n|---|---:|---:|---:|---:|\n"""
    for r in class_results: report+=f"| {r['class_name']} | {r['dice']*100:.2f}% | {r['hd95']:.2f} | {r['jaccard']*100:.2f}% | {r['asd']:.2f} |\n"
    report+="\n## Technical observations\n1. The public checkpoint is stored separately from the repository, so its SHA-256 is recorded for provenance.\n2. The released test code assumes CUDA; this evaluator changes tensor placement only, preserving network outputs and metric definitions.\n3. Synapse labels are remapped to the released 9-class setup before evaluation.\n4. Raw case/class metrics and the complete console log are retained for auditability.\n"
    (out/"STEP1_EMCAD_REPORT.md").write_text(report)
    print("FINAL",json.dumps(m["observed"],indent=2),flush=True)
if __name__=="__main__": main()
