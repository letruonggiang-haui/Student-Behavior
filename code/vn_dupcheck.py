"""Near-duplicate audit of the independent transfer set, the same difference-hash audit already
run on the Kaggle engagement corpus. Reports how many image pairs fall within a 64-bit
difference-hash distance of 4, and how many of those pairs cross identity folders."""
import os, json, itertools
from pathlib import Path
from PIL import Image
import imagehash

root = Path(os.path.expanduser("~/student_behavior/vn_sample"))
items = [(f, f.parent.name, f.name.split("_")[0]) for f in sorted(root.glob("*/*.jpg"))]
h = {f: imagehash.dhash(Image.open(f), hash_size=8) for f, _, _ in items}
pairs = cross = same = 0
near = set()
for (f1, c1, s1), (f2, c2, s2) in itertools.combinations(items, 2):
    if h[f1] - h[f2] <= 4:
        pairs += 1
        near.add(f1); near.add(f2)
        cross += int(s1 != s2)
        same += int(c1 == c2)
out = {"images": len(items), "pairs_dhash_le_4": pairs, "images_in_a_near_duplicate_pair": len(near),
       "pairs_across_identity_folders": cross, "pairs_same_class": same}
json.dump(out, open(os.path.expanduser("~/student_behavior/runs/revision/vn_dupcheck.json"), "w"), indent=1)
print("VN_DUPCHECK", json.dumps(out))
