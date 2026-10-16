import pandas as pd, numpy as np, re
R="E:/datasets/EmotiW_2023_EngageNet/"
def load(split):
    if split=="Test": df=pd.read_csv(R+"test_engagement_labels.csv")
    else: df=pd.read_excel(R+f"{split.lower()}_engagement_labels.xlsx")[["chunk","label"]]
    df["split"]=split
    return df
df=pd.concat([load(s) for s in ["Train","Validation","Test"]],ignore_index=True)
pat=re.compile(r"subject_(\d+)_([a-z0-9]+)_vid_(\d+)_(\d+)\.mp4")
m=df.chunk.str.extract(pat); df["subj"]=m[0].astype(int); df["sid"]=m[1]; df["vid"]=m[2].astype(int); df["k"]=m[3].astype(int)
print("total clips",len(df)); print(df.groupby("split").size().to_string())
print("subjects per split"); print(df.groupby("split").subj.nunique().to_string()); print("subjects total",df.subj.nunique())
# subject overlap between splits
S={s:set(df[df.split==s].subj) for s in ["Train","Validation","Test"]}
print("overlap tr/va",len(S["Train"]&S["Validation"]),"tr/te",len(S["Train"]&S["Test"]),"va/te",len(S["Validation"]&S["Test"]))
print("label dist overall"); print(df.label.value_counts().to_string())
# videos
df["video"]=df.subj.astype(str)+"_"+df.vid.astype(str)
g=df.groupby("video").size(); print("videos",len(g),"clips/video mean %.1f median %.0f max %d"%(g.mean(),g.median(),g.max()))
print("videos per subject", df.groupby("subj").vid.nunique().describe().to_string())
# consecutive agreement within video (sorted by k)
d=df.sort_values(["video","k"]); agree=[];n=0
for v,gg in d.groupby("video"):
    l=gg.label.values; k=gg.k.values
    for i in range(1,len(l)):
        if k[i]==k[i-1]+1: agree.append(l[i]==l[i-1])
print("consecutive same-label prob %.3f (n=%d)"%(np.mean(agree),len(agree)))
# gaps in k (missing chunks)
gaps=sum((gg.k.diff().dropna()!=1).sum() for v,gg in d.groupby("video")); print("index gaps within videos",gaps)
# ordinal mapping & between-subject variance
mp={"Not-Engaged":0,"Barely-engaged":1,"Engaged":2,"Highly-Engaged":3}
e=df[df.label.isin(mp)].copy(); e["y"]=e.label.map(mp)
tot=e.y.var(ddof=0); sm=e.groupby("subj").y.transform("mean"); between=sm.var(ddof=0)
print("between-subject variance share %.3f"%(between/tot))
frac=e.groupby("subj").y.agg(lambda s:s.value_counts(normalize=True).iloc[0]); print("subjects with >=80%% one label: %d / %d"%((frac>=0.8).sum(),len(frac)))
maj=e.groupby("subj").y.agg(lambda s:s.value_counts().index[0]); print("majority label per subject dist"); print(maj.value_counts().sort_index().to_string())
# per-split label dist
print(pd.crosstab(df.split,df.label,normalize="index").round(3).to_string())
# majority class accuracy on test
te=e[e.split=="Test"]; print("test majority-class acc %.3f"%(te.y.value_counts(normalize=True).max()))
# subject-majority oracle: predict each test subject's own majority label
orc=(te.y==te.groupby("subj").y.transform(lambda s:s.value_counts().index[0])).mean(); print("test per-subject-majority oracle acc %.3f"%orc)
df.to_csv("labels_all.csv",index=False)
