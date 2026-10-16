# Extract representative frames (start / middle / end) from every EngageNet clip.
# Output stays local (EULA: no redistribution). Frames resized so the longer side is 640 px.
import cv2, os, sys, time, json
from concurrent.futures import ThreadPoolExecutor
R = "E:/datasets/EmotiW_2023_EngageNet/"
OUT = "E:/datasets/EmotiW_2023_EngageNet_frames/"
POS = {"s": 0.10, "m": 0.50, "e": 0.90}


def work(args):
    split, f = args
    src = R + split + "/" + f; base = OUT + split + "/" + f[:-4]
    if all(os.path.exists(base + f"_{k}.jpg") for k in POS): return (f, "skip")
    cap = cv2.VideoCapture(src)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    if n <= 0: cap.release(); return (f, "nof")
    ok_all = True
    for k, p in POS.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(n * p))))
        ok, img = cap.read()
        if not ok:
            ok_all = False; continue
        h, w = img.shape[:2]; s = 640 / max(h, w)
        if s < 1: img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(base + f"_{k}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cap.release()
    return (f, "ok" if ok_all else "partial", n, fps)


if __name__ == "__main__":
    jobs = []
    for split in ["Validation", "Test", "Train"]:
        os.makedirs(OUT + split, exist_ok=True)
        jobs += [(split, f) for f in sorted(os.listdir(R + split)) if f.endswith(".mp4")]
    t0 = time.time(); stats = {}; meta = {}
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(work, jobs)):
            stats[r[1]] = stats.get(r[1], 0) + 1
            if len(r) == 4: meta[r[0]] = {"frames": r[2], "fps": r[3]}
            if i % 500 == 0: print(i, len(jobs), stats, "%.0fs" % (time.time() - t0), flush=True)
    json.dump(meta, open(OUT + "video_meta.json", "w"))
    print("done", stats, "%.0fs" % (time.time() - t0))
