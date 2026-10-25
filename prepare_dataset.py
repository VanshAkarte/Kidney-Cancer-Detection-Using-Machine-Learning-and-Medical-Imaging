"""
prepare_dataset.py  -  TCIA C4KC-KiTS (DICOM CT + DICOM-SEG)  ->  labelled PNG slices
====================================================================================
Output layout expected by kidney_cancer_pipeline.py:
    data/tumor/<patientID>__<instance>.png   slice contains tumour segmentation
    data/normal/<patientID>__<instance>.png  slice contains kidney but NO tumour/cyst

Labelling rule (derived from the expert segmentations, so labels are ground truth):
    tumor  : tumour pixels  >= --min-px on that slice
    normal : kidney pixels  >= --min-px, tumour pixels < --min-px, cyst pixels < --min-px
    other slices (no kidney, or cyst only) are skipped.

Usage:
    python prepare_dataset.py --src tciaDownload --out data --per-class 12
"""
import argparse, os, re, glob, collections
import numpy as np, pydicom
from PIL import Image


def role_of(label, a):
    l = label.lower()
    if re.search(a.tumor_regex, l): return "tumor"
    if "cyst" in l: return "cyst"
    if re.search(a.kidney_regex, l): return "kidney"
    return "other"


def window(img_hu, wl, ww):
    lo, hi = wl - ww / 2, wl + ww / 2
    return ((np.clip(img_hu, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def scan(root):
    ct, seg = collections.defaultdict(list), []
    for f in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isfile(f): continue
        try: d = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception: continue
        m = getattr(d, "Modality", None)
        if m == "CT": ct[d.SeriesInstanceUID].append(f)
        elif m == "SEG": seg.append(f)
    return ct, seg


def main(a):
    rng = np.random.default_rng(42)
    ct, segs = scan(a.src)
    print(f"Found {len(ct)} CT series and {len(segs)} SEG files")
    rows, seen_labels, done_patients = [], set(), set()
    for sf in sorted(segs):
        s = pydicom.dcmread(sf)
        try: ref = s.ReferencedSeriesSequence[0].SeriesInstanceUID
        except Exception: print("skip (no referenced series):", sf); continue
        if ref not in ct: print("skip (referenced CT not downloaded):", sf); continue

        # CT headers of the referenced series
        info = {}
        for f in ct[ref]:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            info[d.SOPInstanceUID] = dict(path=f, z=float(d.ImagePositionPatient[2]),
                                          inst=int(d.InstanceNumber), pid=str(d.PatientID))
        pid = re.sub(r"[^A-Za-z0-9-]", "-", next(iter(info.values()))["pid"])
        if a.max_patients and pid not in done_patients and len(done_patients) >= a.max_patients: continue
        zs = np.array([v["z"] for v in info.values()]); sops = list(info.keys())

        roles = {int(x.SegmentNumber): role_of(str(x.SegmentLabel), a) for x in s.SegmentSequence}
        seen_labels |= {str(x.SegmentLabel) for x in s.SegmentSequence}
        arr = s.pixel_array
        if arr.ndim == 2: arr = arr[None]
        pf = s.PerFrameFunctionalGroupsSequence
        cnt = collections.defaultdict(lambda: collections.defaultdict(int))
        for i in range(arr.shape[0]):
            role = roles[int(pf[i].SegmentIdentificationSequence[0].ReferencedSegmentNumber)]
            try:
                sop = pf[i].DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
            except Exception:                                   # fallback: match by z position
                z = float(pf[i].PlanePositionSequence[0].ImagePositionPatient[2])
                j = int(np.argmin(np.abs(zs - z)))
                if abs(zs[j] - z) > 1.0: continue
                sop = sops[j]
            if sop in info: cnt[sop][role] += int((arr[i] > 0).sum())

        tum, nor = [], []
        for sop, c in cnt.items():
            if c["tumor"] >= a.min_px: tum.append((sop, c["tumor"]))
            elif c["kidney"] >= a.min_px and c["cyst"] < a.min_px: nor.append((sop, 0))
        if not tum: print(f"{pid}: no tumour slices, skipped"); continue
        done_patients.add(pid)

        for name, lst in (("tumor", tum), ("normal", nor)):
            lst = sorted(lst, key=lambda t: info[t[0]]["z"])
            if len(lst) > a.per_class:                          # evenly spaced sample across the kidney
                lst = [lst[k] for k in np.linspace(0, len(lst) - 1, a.per_class).round().astype(int)]
            os.makedirs(os.path.join(a.out, name), exist_ok=True)
            for sop, px in lst:
                v = info[sop]; d = pydicom.dcmread(v["path"])
                hu = d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) \
                     + float(getattr(d, "RescaleIntercept", 0))
                fn = f"{pid}__{v['inst']:04d}.png"
                Image.fromarray(window(hu, a.wl, a.ww)).save(os.path.join(a.out, name, fn))
                rows.append((pid, fn, name, px))
        print(f"{pid}: tumour slices={sum(r[0]==pid and r[2]=='tumor' for r in rows)}, "
              f"normal slices={sum(r[0]==pid and r[2]=='normal' for r in rows)}")

    print("\nSegment labels seen in SEG files:", sorted(seen_labels))
    print("If tumour/kidney were not recognised, change --tumor-regex / --kidney-regex.")
    import pandas as pd
    df = pd.DataFrame(rows, columns=["patient", "file", "label", "tumor_pixels"])
    df.to_csv("dataset_summary.csv", index=False)
    print(f"\nPatients: {df.patient.nunique()} | slices: {len(df)}\n{df.label.value_counts().to_string()}")
    print("Saved dataset_summary.csv (use it for the Dataset Description section of your report)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder with downloaded DICOM (CT + SEG)")
    p.add_argument("--out", default="data")
    p.add_argument("--per-class", type=int, default=12, help="max slices per class per patient")
    p.add_argument("--min-px", type=int, default=50)
    p.add_argument("--max-patients", type=int, default=0, help="0 = all downloaded")
    p.add_argument("--wl", type=int, default=40); p.add_argument("--ww", type=int, default=400)
    p.add_argument("--tumor-regex", default=r"tumou?r|mass|cancer|neoplasm")
    p.add_argument("--kidney-regex", default=r"kidney|renal")
    main(p.parse_args())
