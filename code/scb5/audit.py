import os, glob, json
from collections import Counter
from PIL import Image
R = os.path.expanduser("~/student_behavior/scb5")
PACKS = {
 "handrise_read_write": ("SCB5-Handrise-Read-write-2024-9-17", ["hand-raising","read","write"]),
 "discuss": ("SCB5-Discuss-2024-9-17", ["discuss"]),
 "bow_turn_head": ("SCB_BowTurnHead_20250509", ["BowHead","TurnHead"]),
 "teacher": ("SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2", None),
}
out={}
for name,(d,names) in PACKS.items():
    base=os.path.join(R,d)
    lab=sorted(glob.glob(base+"/**/labels/**/*.txt",recursive=True))
    lab=[p for p in lab if os.path.basename(p)!="classes.txt"]
    cls=Counter(); per_img=[]; splits=Counter()
    for p in lab:
        splits["val" if "/val/" in p else "train" if "/train/" in p else "?"]+=1
        n=0
        for line in open(p):
            line=line.split()
            if len(line)>=5: cls[int(line[0])]+=1; n+=1
        per_img.append(n)
    imgs=sorted(glob.glob(base+"/**/images/**/*.jpg",recursive=True))+sorted(glob.glob(base+"/**/images/**/*.png",recursive=True))
    res=Counter()
    for p in imgs[::max(1,len(imgs)//200)][:200]:
        try: res[Image.open(p).size]+=1
        except Exception: pass
    tot=sum(per_img)
    out[name]=dict(images=len(imgs),label_files=len(lab),boxes=tot,
        boxes_per_img=round(tot/max(1,len(per_img)),2),
        max_boxes=max(per_img) if per_img else 0,
        empty_imgs=sum(1 for n in per_img if n==0),
        splits=dict(splits), class_counts={ (names[k] if names and k<len(names) else str(k)):v for k,v in sorted(cls.items())},
        top_resolutions=[(f"{w}x{h}",c) for (w,h),c in res.most_common(4)])
print(json.dumps(out,indent=1,ensure_ascii=False))
json.dump(out,open(R+"/audit.json","w"),indent=1,ensure_ascii=False)
