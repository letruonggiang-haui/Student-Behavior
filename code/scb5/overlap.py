import os, glob, hashlib, json
from collections import Counter, defaultdict
R = os.path.expanduser("~/student_behavior/scb5")
PACKS = {
 "hrw": "SCB5-Handrise-Read-write-2024-9-17",
 "dis": "SCB5-Discuss-2024-9-17",
 "bth": "SCB_BowTurnHead_20250509",
 "tea": "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
}
def md5(p):
    h=hashlib.md5()
    with open(p,"rb") as f:
        while True:
            b=f.read(1<<20)
            if not b: break
            h.update(b)
    return h.hexdigest()
names={}; hashes={}
for k,d in PACKS.items():
    imgs=glob.glob(os.path.join(R,d)+"/**/images/**/*.jpg",recursive=True)+glob.glob(os.path.join(R,d)+"/**/images/**/*.png",recursive=True)
    names[k]=set(os.path.basename(p) for p in imgs)
    hashes[k]={}
    for p in imgs: hashes[k].setdefault(md5(p),[]).append(p)
    print(k,"imgs",len(imgs),"uniq names",len(names[k]),"uniq md5",len(hashes[k]),flush=True)
ks=list(PACKS)
print("--- pairwise name overlap ---")
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        a,b=ks[i],ks[j]
        print(f"{a} x {b}: names {len(names[a]&names[b])}  md5 {len(set(hashes[a])&set(hashes[b]))}")
json.dump({k:sorted(hashes[k]) for k in ks},open(R+"/hashes.json","w"))
