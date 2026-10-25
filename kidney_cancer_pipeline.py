"""
Kidney Cancer Detection using Machine Learning and Medical Imaging (TCIA CT)
=============================================================================
Pipeline:  data -> dedupe -> patient-level split -> preprocessing (CLAHE)
        -> BASELINES: (A) handcrafted features + Logistic Regression
                      (B) fine-tuned ResNet18 CNN
        -> PROPOSED : hybrid  (frozen CNN deep features + handcrafted texture
                      features) -> feature selection (tuned k) -> stacking ensemble
        -> evaluation, plots, Grad-CAM explainability

EXPECTED DATA LAYOUT (2D CT slices exported as PNG/JPG):
    data/normal/<patientID>__<anything>.png
    data/tumor/<patientID>__<anything>.png
The text before "__" is the patient ID. It is used to split by PATIENT so
slices of one patient never appear in both train and test (avoids leakage).

If you only have DICOM, use dicom_to_png() at the bottom to make slices, and
build labels from the collection's clinical/segmentation files.

USAGE (Colab):
    !pip install pydicom scikit-image seaborn
    !python kidney_cancer_pipeline.py --data data --out results --epochs 8
    !python kidney_cancer_pipeline.py --demo      # synthetic smoke test
"""
import argparse, hashlib, os, glob, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
from scipy.stats import skew, kurtosis, entropy
from skimage import exposure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
import torch, torch.nn as nn
from torchvision import models
warnings.filterwarnings("ignore")
SEED = 42; IMG = 224
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- demo data
def make_demo(root, n_pat=24, per_pat=8):
    rng = np.random.default_rng(0)
    for cls in ("normal", "tumor"):
        os.makedirs(f"{root}/{cls}", exist_ok=True)
    for p in range(n_pat):
        cls = "tumor" if p % 2 else "normal"
        for s in range(per_pat):
            im = rng.normal(90, 25, (IMG, IMG)).clip(0, 255)
            yy, xx = np.ogrid[:IMG, :IMG]
            im[(yy - 112) ** 2 + (xx - 90) ** 2 < 40 ** 2] += 40          # "kidney"
            if cls == "tumor":
                cx, cy, r = rng.integers(80, 140), rng.integers(80, 140), rng.integers(10, 22)
                im[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] -= 45        # "lesion"
            Image.fromarray(im.clip(0, 255).astype(np.uint8)).save(f"{root}/{cls}/P{p:03d}__s{s}.png")


# ---------------------------------------------------------------- data + preprocessing
def load_index(root):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}"))
    assert len(classes) == 2, f"Expected 2 class folders, found {classes}"
    rows, seen = [], set()
    for ci, c in enumerate(classes):
        for f in glob.glob(f"{root}/{c}/*"):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")): continue
            h = hashlib.md5(open(f, "rb").read()).hexdigest()
            if h in seen: continue                      # remove duplicate images
            seen.add(h)
            pid = os.path.basename(f).split("__")[0]
            rows.append((f, ci, pid))
    return pd.DataFrame(rows, columns=["path", "label", "patient"]), classes


def preprocess(path):
    """grayscale -> resize -> CLAHE contrast enhancement -> float [0,1]"""
    im = Image.open(path).convert("L").resize((IMG, IMG))
    a = np.asarray(im, dtype=np.float32) / 255.0
    return exposure.equalize_adapthist(a, clip_limit=0.02).astype(np.float32)


# ---------------------------------------------------------------- features
def handcrafted(a):
    u8 = (a * 255).astype(np.uint8)
    h, _ = np.histogram(u8, bins=32, range=(0, 255), density=True)
    f = [a.mean(), a.std(), skew(a.ravel()), kurtosis(a.ravel()), entropy(h + 1e-9),
         *np.percentile(a, [5, 25, 50, 75, 95])]
    q = (u8 // 4).astype(np.uint8)                       # 64 grey levels
    g = graycomatrix(q, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 64, symmetric=True, normed=True)
    for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        f += list(graycoprops(g, p).mean(axis=1))        # one value per distance
    lbp = local_binary_pattern(u8, 8, 1, "uniform")
    f += list(np.histogram(lbp, bins=10, range=(0, 10), density=True)[0])
    return np.array(f, dtype=np.float32)


def to_tensor(a):
    t = torch.from_numpy(a)[None].repeat(3, 1, 1)
    m = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; s = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (t - m) / s


def deep_features(arrs):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity(); net.eval().to(DEV)
    out = []
    with torch.no_grad():
        for i in range(0, len(arrs), 32):
            b = torch.stack([to_tensor(x) for x in arrs[i:i + 32]]).to(DEV)
            out.append(net(b).cpu().numpy())
    return np.vstack(out)


# ---------------------------------------------------------------- metrics / plots
def metrics(name, y, prob, thr=0.5):
    p = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(Model=name, Accuracy=accuracy_score(y, p), Precision=precision_score(y, p, zero_division=0),
                Recall_Sensitivity=recall_score(y, p), Specificity=tn / (tn + fp + 1e-9),
                F1=f1_score(y, p), ROC_AUC=roc_auc_score(y, prob)), confusion_matrix(y, p)


# ---------------------------------------------------------------- CNN baseline + Grad-CAM
def finetune_cnn(Xtr, ytr, gtr, epochs):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, 2); net.to(DEV)
    va_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(Xtr, ytr, gtr))
    tr_i, va_i = va_idx
    w = torch.tensor(len(ytr) / (2 * np.bincount(ytr)), dtype=torch.float32).to(DEV)  # class weights
    lossf = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(net.parameters(), 1e-4)
    T = lambda idx: torch.stack([to_tensor(Xtr[i]) for i in idx])
    best, best_state = -1, None
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(tr_i)
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16]; xb = T(idx).to(DEV)
            if np.random.rand() < .5: xb = torch.flip(xb, [3])            # simple augmentation
            loss = lossf(net(xb), torch.tensor(ytr[idx]).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        pv = predict_cnn(net, [Xtr[i] for i in va_i])
        auc = roc_auc_score(ytr[va_i], pv) if len(set(ytr[va_i])) > 1 else 0
        print(f"  epoch {ep + 1}/{epochs}  val AUC={auc:.3f}")
        if auc >= best: best, best_state = auc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); return net


def predict_cnn(net, arrs):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(arrs), 32):
            b = torch.stack([to_tensor(x) for x in arrs[i:i + 32]]).to(DEV)
            out.append(torch.softmax(net(b), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def gradcam(net, arr):
    net.eval(); feats, grads = [], []
    h1 = net.layer4.register_forward_hook(lambda m, i, o: feats.append(o))
    h2 = net.layer4.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    x = to_tensor(arr)[None].to(DEV); out = net(x); net.zero_grad(); out[0, 1].backward()
    h1.remove(); h2.remove()
    w = grads[0].mean((2, 3), keepdim=True); cam = torch.relu((w * feats[0]).sum(1))[0].detach().cpu().numpy()
    cam = np.array(Image.fromarray(cam).resize((IMG, IMG), Image.BILINEAR))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)


# ---------------------------------------------------------------- main
def main(a):
    os.makedirs(a.out, exist_ok=True)
    if a.demo: make_demo("demo_data"); a.data = "demo_data"; a.epochs = 2
    df, classes = load_index(a.data)
    print(f"Images after de-duplication: {len(df)} | patients: {df.patient.nunique()} | classes: {classes}")
    print(df.groupby("label").size().rename(index=dict(enumerate(classes))))

    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(df, df.label, df.patient))
    arrs = [preprocess(p) for p in df.path]; y = df.label.values; g = df.patient.values
    print("Extracting features ...")
    H = np.vstack([handcrafted(x) for x in arrs]); D = deep_features(arrs)
    X = np.hstack([D, H]); nD = D.shape[1]
    names = [f"cnn_{i}" for i in range(nD)] + [f"hand_{i}" for i in range(H.shape[1])]
    results, cms, probs = [], {}, {}

    # --- Baseline A: handcrafted + LR
    lr = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    lr.fit(H[tr], y[tr]); probs["Baseline A: Handcrafted+LR"] = lr.predict_proba(H[te])[:, 1]

    # --- Baseline B: fine-tuned CNN
    print("Fine-tuning ResNet18 baseline ...")
    cnn = finetune_cnn(np.stack(arrs).astype(np.float32)[tr], y[tr], g[tr], a.epochs)
    probs["Baseline B: Fine-tuned ResNet18"] = predict_cnn(cnn, [arrs[i] for i in te])

    # --- Proposed: hybrid features -> tuned feature selection -> stacking ensemble
    def build(k):
        stack = StackingClassifier(
            estimators=[("rf", RandomForestClassifier(300, class_weight="balanced", random_state=SEED)),
                        ("svm", SVC(probability=True, class_weight="balanced", random_state=SEED)),
                        ("gb", GradientBoostingClassifier(random_state=SEED))],
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            cv=3, n_jobs=-1)
        return Pipeline([("s", StandardScaler()), ("fs", SelectKBest(mutual_info_classif, k=k)), ("m", stack)])
    print("Tuning number of selected features (patient-grouped CV) ...")
    ks, cv_auc = [k for k in (16, 32, 64, 128) if k <= X.shape[1]], {}
    for k in ks:
        sc = []
        for a_, b_ in GroupKFold(3).split(X[tr], y[tr], g[tr]):
            if len(set(y[tr][b_])) < 2: continue
            m = build(k).fit(X[tr][a_], y[tr][a_]); sc.append(roc_auc_score(y[tr][b_], m.predict_proba(X[tr][b_])[:, 1]))
        cv_auc[k] = np.mean(sc) if sc else 0; print(f"  k={k}: CV AUC={cv_auc[k]:.3f}")
    bestk = max(cv_auc, key=cv_auc.get); print("Best k =", bestk)
    prop = build(bestk).fit(X[tr], y[tr]); probs["PROPOSED: Hybrid+FS+Stacking"] = prop.predict_proba(X[te])[:, 1]

    # --- evaluate
    for n, p in probs.items():
        r, cm = metrics(n, y[te], p); results.append(r); cms[n] = cm
    res = pd.DataFrame(results).round(4); res.to_csv(f"{a.out}/results.csv", index=False); print("\n", res.to_string(index=False))

    # --- plots
    plt.figure(figsize=(6, 5))
    for n, p in probs.items():
        fpr, tpr, _ = roc_curve(y[te], p); plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y[te], p):.2f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
    plt.title("ROC curves (test set)"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{a.out}/roc.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for axi, (n, cm) in zip(ax, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axi, cbar=False)
        axi.set_title(n, fontsize=8); axi.set_xlabel("Predicted"); axi.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(f"{a.out}/confusion_matrices.png", dpi=150); plt.close()

    res.set_index("Model")[["Accuracy", "Recall_Sensitivity", "Specificity", "F1", "ROC_AUC"]].plot.bar(figsize=(9, 4))
    plt.ylim(0, 1.05); plt.xticks(rotation=15, fontsize=7); plt.title("Model comparison"); plt.tight_layout()
    plt.savefig(f"{a.out}/model_comparison.png", dpi=150); plt.close()

    # feature importance of selected features (RF inside stack) + CNN-vs-handcrafted share
    sel = prop.named_steps["fs"].get_support(indices=True)
    rf = prop.named_steps["m"].named_estimators_["rf"]
    imp = pd.Series(rf.feature_importances_, index=[names[i] for i in sel]).sort_values(ascending=False)
    imp.head(15).iloc[::-1].plot.barh(figsize=(6, 5), title="Top selected features (RF importance)")
    plt.tight_layout(); plt.savefig(f"{a.out}/feature_importance.png", dpi=150); plt.close()
    share = imp.groupby(lambda n: "Deep CNN" if n.startswith("cnn") else "Handcrafted texture").sum()
    print("\nImportance share by feature type:\n", (share / share.sum()).round(3))

    # correlation heatmap of handcrafted features
    sns.heatmap(pd.DataFrame(H[:, :20]).corr(), cmap="coolwarm", center=0); plt.title("Handcrafted feature correlation (first 20)")
    plt.tight_layout(); plt.savefig(f"{a.out}/correlation.png", dpi=150); plt.close()

    # Grad-CAM on some test tumor images (explainability)
    ti = [i for i in te if y[i] == 1][:4]
    if ti:
        fig, ax = plt.subplots(2, len(ti), figsize=(3 * len(ti), 6), squeeze=False)
        for j, i in enumerate(ti):
            cam = gradcam(cnn, arrs[i]); ax[0, j].imshow(arrs[i], cmap="gray"); ax[0, j].axis("off")
            ax[1, j].imshow(arrs[i], cmap="gray"); ax[1, j].imshow(cam, cmap="jet", alpha=.4); ax[1, j].axis("off")
        ax[0, 0].set_title("CT slice", fontsize=9); ax[1, 0].set_title("Grad-CAM", fontsize=9)
        plt.tight_layout(); plt.savefig(f"{a.out}/gradcam.png", dpi=150); plt.close()
    print(f"\nAll outputs saved in ./{a.out}")


# ---------------------------------------------------------------- DICOM helper
def dicom_to_png(dicom_dir, out_dir, patient_id, label_name, wl=40, ww=400):
    """Convert a CT series to kidney-window PNG slices named <patient>__<n>.png"""
    import pydicom
    os.makedirs(f"{out_dir}/{label_name}", exist_ok=True)
    for n, f in enumerate(sorted(glob.glob(f"{dicom_dir}/**/*.dcm", recursive=True))):
        d = pydicom.dcmread(f); hu = d.pixel_array * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))
        lo, hi = wl - ww / 2, wl + ww / 2
        Image.fromarray(((hu.clip(lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)).save(f"{out_dir}/{label_name}/{patient_id}__{n}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--demo", action="store_true")
    main(ap.parse_args())
