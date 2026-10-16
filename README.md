# The accuracy cost of adding a student-engagement head to a classroom behaviour detector

Code, split definitions, audit scripts and per-seed result files for the paper

> Le T.G., Nguyen T.V., Nguyen N.S. *The accuracy cost of adding a student-engagement head to a classroom behaviour detector.* Manuscript under review, 2026.

The paper trains a YOLO11 detector with a pooled engagement head on the face-mosaiced public release of SCB-Dataset5 and on EngageNet frames, and measures what the second head costs the detector, how classroom-trained features transfer to an independent classroom dataset, and what four public benchmarks actually measure. The numbers in the manuscript are computed from the files in `results/`; the table under "Which file supports which section" maps them to the paper.

No dataset is redistributed here. See "Data" below.

## Layout

```
code/
  train_scb5.py            single-task detector (YOLO11 n/s/m/l/x, RT-DETR-l) on an SCB5 pack
  mtl.py                   MTLDetectionModel: YOLO11 + engagement head (GAP + MLP on the stride-32 map),
                           MTLTrainer that interleaves engagement mini-batches with detection batches
  train_mtl.py             multi-task training (--eng-classes 2 for the Kaggle set, 4 for EngageNet)
  engcal2.py               refit of the engagement head on frozen cached features, class-balanced CE,
                           model selection by balanced accuracy on the validation participants
  evaluate_scb5.py         aggregate detection evaluation (3 seeds, per-class AP, Pareto points)
  cross_country_scb5.py    leave-one-folder-out linear probe on the independent classroom set
  eval_engagenet.py        balanced accuracy / macro-F1 with 2,000-resample participant bootstrap
  eng_e2e_ref.py           end-to-end fine-tuned YOLO11s classifier at 224 px (upper reference)
  build_engagenet_frames.py  ImageFolder from one middle frame per clip, official subject split, SNP dropped
  engagenet/extract_frames.py  frame extraction from EngageNet clips (run on the licensee's machine only)
  fps_final.py             batch-1 throughput on an idle GPU, 200 frames after 20 warm-up
  run_scb5.sh              main detection pipeline (MTL x3 seeds, det-only x3, RT-DETR-l, YOLO11n/m, BTH pack)
  run_ablation.sh          same models on the non-mosaiced images (cost of face mosaicing)
  run_sizematch.sh         YOLO11 m/l/x on the non-mosaiced images (size-matched comparison)
  run_mtl_l.sh             multi-task YOLO11l x3 seeds
  run_engagenet.sh         all EngageNet configurations (frozen references, MTL s/l, end-to-end)
  eng_dupcheck.py, eng_clusters.py, b2_check.py, b2_clean.py, eng_check.py,
  eng_raw_head.py, eng_unfreeze.py   diagnostics behind Section 4.5 of the paper
  run_revision_A.sh .. run_revision_D.sh   revision runs: saved per-frame predictions, batch-norm
                           recalibration, throughput re-measurement, extra seeds and controls
  bn_recal_eval.py, bn_recal_aug.py   detection-side batch-norm recalibration and its control
  fps_v2.py                throughput of every configuration under one timed region
  probe_v2.py              transfer probe over all seeds, with a within-domain SCB5 crop control
  paired_boot.py           paired participant-bootstrap intervals for differences in balanced accuracy
  eng_e2e_preds.py         per-frame predictions of the end-to-end classifier (kept on the licensee's machine)
  scb_dupcheck.py, vn_dupcheck.py   near-duplicate audits of the SCB5 split and the transfer set
  scb5/
    dl.sh, dl2.sh          download of the public face-mosaiced release from Hugging Face
    build_scb5_mosaic.py   YOLO sets from the mosaiced release (labels and split byte-identical to the original)
    build_nomosaic.py      the same split from the non-mosaiced pack
    build_scb5.py, audit.py, overlap.py, merge_plan.py, label_stats.py, boxsize.py
                           pack inventory: images shared across packs, boxes per image, object sizes
    dedup_audit.py         duplicate label rows per pack (Ultralytics drops the rows silently)
    fp_analysis.py, fp_analysis2.py   false positives that land on boxes labelled by another pack
  yolov7_patch/
    prep_yolov7.sh, run_v7.sh      reproduction of the published YOLOv7 reference on the same pack
    eval_v7subset.py               score our models on exactly the validation subset the YOLOv7 repository keeps
    yolov7_compat.patch            changes needed to run the YOLOv7 repository on torch 2.13 / numpy 2.5
    yolov7/torch_compat.py, yolov7/data/scb5_hrw_orig.yaml
results/
  detection/    per-seed detection results, Pareto points, FPS, size-matched runs, first transfer probe
  engagement/   EngageNet references, MTL heads (jointly trained and refitted), end-to-end classifier,
                Kaggle-set diagnostics
  revision/     revision runs: engagement evaluations per seed, batch-norm recalibration, throughput,
                transfer probe, paired bootstrap, per-participant accuracy, near-duplicate audits
  audits/       label-duplication audit, mosaic build record, missing-label analysis
  yolov7/       results.txt, hyp.yaml and opt.yaml of the two YOLOv7 reproduction runs
  curves/       per-epoch results.csv of the training runs
```

## Which file supports which section

Paths are relative to `results/`; section numbers are those of the manuscript.

| Manuscript | Files |
|---|---|
| Section 4.1 (RQ1), cost of the engagement head | `detection/final_eval_scb5.json`, `mtl_l_eval.json`, `det_scb5_s{0,1,2}.json`, `det_l_s{0,1,2}.json`, `fps_final.json`; `engagement/mtl_en_{s,l}_s0_det.json`; `revision/fps_v2.json`, `bn_recal.json`, `bn_recal_aug.json`; `curves/*/results.csv` |
| Section 4.2 (RQ2), published reference | `detection/det_scb5orig_s{0,1,2}.json`, `v7subset_comparison.json`, `sizematch_subset.json`, `y11{m,l,x}_orig.json`; `yolov7/` |
| Section 4.3 (RQ3), transfer | `detection/cross_country_scb5.json`; `revision/probe_v2.json`, `scb_dupcheck.json`, `vn_dupcheck.json` |
| Section 4.4 (RQ4), engagement | `engagement/eng_ref_coco_s.json`, `eng_ref_det_s.json`, `mtl_en_{s,l}_s0_{rawhead,cal}.json`, `eng_e2e_cls_s.json`; `revision/eng_*.json`, `paired_boot.json`, `per_participant_acc.json` |
| Section 4.5, benchmark validity | `audits/*.json`; `detection/det_scb5orig_s{0,1,2}.json`, `v7subset_comparison.json`; `engagement/engagement2_dupcheck.json`, `engagement_final.json` |

The YOLOv7 mAP values quoted in the paper (74.54 and 74.33) are the best-checkpoint test results printed at the end of each run; `results/yolov7/*/results.txt` holds the per-epoch validation curve of the same runs.

## Data

- **SCB-Dataset5, face-mosaiced public release**: https://huggingface.co/datasets/wintonYF/SCB-public-mosaic (academic, non-commercial). `code/scb5/dl2.sh` downloads it; `build_scb5_mosaic.py` builds the YOLO sets. The Handrise-Read-write pack is the benchmark used in the paper. The non-mosaiced packs used for the mosaicing ablation and the YOLOv7 reproduction are obtained from the SCB authors.
- **EngageNet**: available from its authors under the EngageNet Dataset User Agreement (https://sites.google.com/view/emotiw2023). Nothing derived from it is in this repository: no frames, no labels, no features, no weights. `extract_frames.py` and `build_engagenet_frames.py` reproduce the frame set on a licensee's machine. Test labels were obtained from the dataset authors.
- **Classroom Student Behaviors** (independent classroom set, transfer probe): https://www.kaggle.com/datasets/phamluhuynhmai/classroom-student-behaviors
- **Kaggle "Student engagement" set** (case study only): https://www.kaggle.com/datasets/joyee19/studentengagement

Trained weights are not distributed because they were trained on data whose licences do not allow redistribution of derived models.

## Environment

Python 3.13.7, PyTorch 2.13.0, Ultralytics 8.4.98, CUDA 13.0, one NVIDIA L4 (24 GB). Pinned versions are in `requirements.txt`. The shell scripts assume the layout `~/student_behavior/{code,work,runs,scb5,scb_mosaic}` and a virtual environment at `~/student_behavior/venv`; edit the variables at the top of each script for another layout.

## Reproducing the paper

1. Download the mosaiced release and build the packs: `code/scb5/dl2.sh`, then `python code/scb5/build_scb5_mosaic.py`. For the mosaicing ablation and the YOLOv7 reproduction, place the non-mosaiced packs under `~/student_behavior/scb5/` and run `build_nomosaic.py`.
2. Detection: `bash code/run_scb5.sh`, then `run_ablation.sh`, `run_sizematch.sh`, `run_mtl_l.sh`. Throughput: `python code/fps_final.py`, and `python code/fps_v2.py` for the single-timed-region measurement.
3. YOLOv7 reference: clone https://github.com/WongKinYiu/yolov7, apply `code/yolov7_patch/yolov7_compat.patch`, copy the two files under `yolov7/`, run `prep_yolov7.sh` and `run_v7.sh`. Score our models on the YOLOv7 validation subset with `code/yolov7_patch/eval_v7subset.py`.
4. Engagement: extract EngageNet frames with `code/engagenet/extract_frames.py`, build the ImageFolder with `code/build_engagenet_frames.py`, run `bash code/run_engagenet.sh`.
5. Transfer probe: `python code/cross_country_scb5.py` and `python code/probe_v2.py` on one mid-sequence frame per clip of the independent classroom set (cap 150 clips per class).
6. Revision runs: `bash code/run_revision_A.sh` to `run_revision_D.sh`, then `python code/paired_boot.py`.

Training the multi-task YOLO11l on the L4 takes about two hours per run under the settings in the scripts; the whole pipeline is a few GPU-days.

## Licence

GPL-3.0 (see `LICENSE`). The training code calls the Ultralytics package, which is released under AGPL-3.0; use of the two together is subject to that licence as well. Result files in `results/` may be reused with attribution to the paper.
