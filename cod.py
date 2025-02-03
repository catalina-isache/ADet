import os
import random
import numpy as np
import networkx as nx
import dgl
import torch
import scipy.sparse as sp
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
import dgl.function as fn
import torch.nn as nn
import torch.optim as optim
import time

def load_mat_as_dgl_nx(mp="acm.mat"):
    dt = sio.loadmat(mp)
    if 'Network' in dt: ad = dt['Network']
    elif 'A' in dt: ad = dt['A']
    else: raise ValueError("No adjacency in .mat")
    if 'Attributes' in dt: ft = dt['Attributes']
    elif 'X' in dt: ft = dt['X']
    else: raise ValueError("No features in .mat")
    if 'Label' in dt: al = dt['Label']
    elif 'gnd' in dt: al = dt['gnd']
    else: raise ValueError("No label in .mat")
    ad = sp.csr_matrix(ad)
    ft = sp.lil_matrix(ft)
    rs = np.array(ft.sum(1))
    ri = np.power(rs, -1).flatten()
    ri[np.isinf(ri)] = 0.
    rmi = sp.diags(ri)
    fn_ = rmi.dot(ft).todense()
    s, d = ad.nonzero()
    g = dgl.graph((s,d))
    g = dgl.add_self_loop(g)
    al = np.squeeze(np.array(al, dtype=int))
    ng = nx.Graph()
    for ss,dd in zip(s,d):
        if not ng.has_edge(ss,dd):
            ng.add_edge(ss,dd, weight=1.0)
    return g, np.array(fn_), al, ng

def set_rs(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    os.environ["PYTHONHASHSEED"] = str(sd)

def rsc(x):
    return (x+1)/2

def agg(g, ft, k):
    dg = g.in_degrees().float().clamp(min=1)
    nm = torch.pow(dg, -0.5).unsqueeze(1).to(ft.device)
    o = ft
    for _ in range(k):
        o = o * nm
        g.ndata['h'] = o
        g.update_all(fn.copy_u('h','m'), fn.sum('m','h'))
        o = g.ndata.pop('h')
        o = o * nm
    return o

def gd(g, k):
    nn_ = g.num_nodes()
    em = torch.eye(nn_, dtype=torch.float32, device=g.device)
    a = agg(g, em, k)
    return torch.diag(a)

class Disc(nn.Module):
    def __init__(self, ni, nh):
        super().__init__()
        self.fg = nn.Linear(ni, nh)
        self.fn = nn.Linear(ni, nh)
    def forward(self, fts, sumy):
        c = nn.functional.cosine_similarity(self.fn(fts), self.fg(sumy))
        return -1*c.unsqueeze(0)

class Mod(nn.Module):
    def __init__(self, g, ni, nh, k=2):
        super().__init__()
        self.g = g
        self.disc = Disc(ni, nh)
    def forward(self, fx, fy):
        return self.disc(fx.detach(), fy.detach())

class DL:
    def __init__(self, g, fts, k=2):
        self.g = g
        self.en = fts
        dv = fts.device
        n = g.num_nodes()
        self.lz = torch.zeros(1, n, device=dv)
        self.lo = torch.ones(1, n, device=dv)
        self.k = k
        with torch.no_grad():
            wd = gd(g, k)
            af = agg(g, fts, k)
            fw = (fts.T * wd).T
            self.eg = af - fw
    def get_data(self):
        ep = self.en
        egp = self.eg
        i2 = torch.randperm(ep.size(0))
        en = ep[i2]
        ega = egp[i2]
        return ep, en, egp, ega

def train_pm(args, ld, md, opt, lf):
    bl = 1e9
    be = -1
    ne = args["num_epoch"]
    a = args["alpha"]
    gm = args["gamma"]
    dv = next(md.parameters()).device
    for e in range(ne):
        md.train()
        ep, en, egp, ega = ld.get_data()
        ep = ep.to(dv)
        en = en.to(dv)
        egp = egp.to(dv)
        ega = ega.to(dv)
        sp = rsc(md(ep, egp))
        sa = rsc(md(ep, ega))
        sn = rsc(md(ep, en))
        lp = lf(sp, ld.lz)
        la = lf(sa, ld.lo)
        ln = lf(sn, ld.lo)
        ls = lp + a*la + gm*ln
        opt.zero_grad()
        ls.backward()
        opt.step()
        if lp.item() < bl:
            bl = lp.item()
            be = e
            torch.save(md.state_dict(), "./temp_prem.ckpt")

def eval_pm(ld, md, al):
    dv = next(md.parameters()).device
    ep = ld.en.to(dv)
    egp = ld.eg.to(dv)
    md.eval()
    with torch.no_grad():
        sc = md(ep, egp).cpu().numpy()[0]
    ac = roc_auc_score(al, sc)
    return sc, ac

def run_pm(g, fts, al, cf):
    set_rs(cf["seed"])
    dv = torch.device(cf["device"] if torch.cuda.is_available() else "cpu")
    ft_t = torch.FloatTensor(fts).to(dv)
    g = g.to(dv)
    ld = DL(g, ft_t, k=cf["k"])
    md = Mod(g, ni=fts.shape[1], nh=cf["n_hidden"], k=cf["k"]).to(dv)
    opt = optim.Adam(md.parameters(), lr=cf["lr"], weight_decay=0.0)
    lf = nn.BCELoss()
    train_pm(cf, ld, md, opt, lf)
    md.load_state_dict(torch.load("./temp_prem.ckpt"))
    sc, av = eval_pm(ld, md, al)
    return sc, av

def cef(ng):
    for nd in ng.nodes():
        neighs = list(ng[nd])
        sn = [nd] + neighs
        sg = ng.subgraph(sn)
        dg = len(neighs)
        es = sg.number_of_edges()
        ws = 0.0
        for (u,v,d) in sg.edges(data=True):
            ws += d.get("weight", 1.0)
        idx_map = {x: i for i,x in enumerate(sn)}
        A = np.zeros((len(sn), len(sn)))
        for (u,v,d) in sg.edges(data=True):
            A[idx_map[u], idx_map[v]] = d.get("weight", 1.0)
            A[idx_map[v], idx_map[u]] = d.get("weight", 1.0)
        lm = 0.0
        if len(sn)>1:
            try:
                vals = np.linalg.eigvals(A)
                lm = np.max(np.abs(vals))
            except:
                lm = 0.0
        ng.nodes[nd]["neighbors"] = dg
        ng.nodes[nd]["edges"] = es
        ng.nodes[nd]["weight"] = ws
        ng.nodes[nd]["eigenvalue"] = lm

def compute_oddball_scores(ng, n_neighbors=2000):
    nl = list(ng.nodes())
    N = np.array([ng.nodes[n]["neighbors"] for n in nl])
    E = np.array([ng.nodes[n]["edges"] for n in nl])
    W = np.array([ng.nodes[n]["weight"] for n in nl])
    w_ = np.zeros(len(nl))
    for i,node in enumerate(nl):
        mw = 0.0
        for nb in ng[node]:
            ew = ng[node][nb].get("weight", 1.0)
            mw = max(mw, ew)
        w_[i] = mw
    def fit_power_law(X, Y):
        msk = (X>0)&(Y>0)
        iv = np.where(msk)[0]
        sc = np.zeros(len(X), dtype=float)
        if len(iv)>1:
            lx = np.log(X[iv]).reshape(-1,1)
            ly = np.log(Y[iv]).reshape(-1,1)
            rg = LinearRegression()
            rg.fit(lx, ly)
            py = np.exp(rg.predict(lx)).flatten()
            ay = Y[iv]
            dl = []
            for a, p in zip(ay, py):
                val = (max(a,p)/min(a,p)) * np.log(abs(a-p)+1)
                dl.append(val)
            dl = np.array(dl).reshape(-1,1)
            dl = MinMaxScaler().fit_transform(dl).flatten()
            for i2, rix in enumerate(iv):
                sc[rix] = dl[i2]
        return sc, iv
    cs_base, cs_valid = fit_power_law(N, E)
    hv_base, hv_valid = fit_power_law(E, W)
    dp_base, dp_valid = fit_power_law(W, w_)
    def apply_lof(X, Y, iv):
        ls = np.zeros(len(nl), dtype=float)
        if len(iv)>1:
            sx = np.column_stack((X[iv], Y[iv]))
            lf = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(iv)-1), n_jobs=-1)
            lf.fit_predict(sx)
            vs = -lf.negative_outlier_factor_
            vs = MinMaxScaler().fit_transform(vs.reshape(-1,1)).flatten()
            for i2, rix in enumerate(iv):
                ls[rix] = vs[i2]
        return ls
    cs_lof = apply_lof(N, E, cs_valid)
    hv_lof = apply_lof(E, W, hv_valid)
    dp_lof = apply_lof(W, w_, dp_valid)
    def scale_to_01(sc):
        if np.all(sc==0):
            return sc
        return MinMaxScaler().fit_transform(sc.reshape(-1,1)).flatten()
    s1 = scale_to_01(cs_base)
    s2 = scale_to_01(cs_lof)
    s3 = scale_to_01(hv_base)
    s4 = scale_to_01(hv_lof)
    s5 = scale_to_01(dp_base)
    s6 = scale_to_01(dp_lof)
    fs = np.maximum.reduce([s1,s2,s3,s4,s5,s6])
    fd = {}
    for i,n in enumerate(nl):
        fd[n] = fs[i]
    return fd

def scale_array(arr):
    arr_2d = arr.reshape(-1,1)
    return MinMaxScaler().fit_transform(arr_2d).flatten()

def cm(yt, yp, pc=5.9):
    th = np.percentile(yp, 100 - pc)
    yb = (yp >= th).astype(int)
    tn = sum((yt==0)&(yb==0))
    fp = sum((yt==0)&(yb==1))
    fn = sum((yt==1)&(yb==0))
    tp = sum((yt==1)&(yb==1))
    acc = (tp+tn)/(tp+tn+fp+fn)
    pre = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    f1s = 2*(pre*rec)/(pre+rec) if (pre+rec)>0 else 0
    auc_ = roc_auc_score(yt, yp)
    return {'accuracy': acc,'precision': pre,'recall': rec,'f1': f1s,'auc': auc_,
            'tp': tp,'tn': tn,'fp': fp,'fn': fn}

def md(mf="acm.mat"):
    g, fts, al, ng = load_mat_as_dgl_nx(mf)
    n_ = g.num_nodes()
    pcu = {"device":"cuda","seed":1,"num_epoch":1500,"alpha":0.3,"gamma":0.4,
           "lr":0.0005,"n_hidden":128,"k":2}
    stc = time.time()
    prem_sc, prem_auc = run_pm(g, fts, al, pcu)
    etc = time.time()
    ctime = etc - stc
    pm_m = cm(al, prem_sc)
    cef(ng)
    sto = time.time()
    odd_sd = compute_oddball_scores(ng, n_neighbors=1)
    eto = time.time()
    obt = eto - sto
    odd_arr = np.zeros(n_)
    for i in range(n_):
        odd_arr[i] = odd_sd.get(i,0.0)
    odd_m = cm(al, odd_arr)
    print(mf)
    print("\nExecution Summary:")
    print(f"PREM (CUDA) Time: {ctime:.4f} seconds")
    print(f"OddBall Time: {obt:.4f} seconds")
    print("\nPREM Metrics:")
    print(f"AUC: {pm_m['auc']:.4f}")
    print(f"Accuracy: {pm_m['accuracy']:.4f}")
    print(f"F1 Score: {pm_m['f1']:.4f}")
    print(f"Precision: {pm_m['precision']:.4f}")
    print(f"Recall: {pm_m['recall']:.4f}")
    print("Confusion Matrix:")
    print(f"TP: {pm_m['tp']}, TN: {pm_m['tn']}")
    print(f"FP: {pm_m['fp']}, FN: {pm_m['fn']}")
    print("\nOddBall Metrics:")
    print(f"AUC: {odd_m['auc']:.4f}")
    print(f"Accuracy: {odd_m['accuracy']:.4f}")
    print(f"F1 Score: {odd_m['f1']:.4f}")
    print(f"Precision: {odd_m['precision']:.4f}")
    print(f"Recall: {odd_m['recall']:.4f}")
    print("Confusion Matrix:")
    print(f"TP: {odd_m['tp']}, TN: {odd_m['tn']}")
    print(f"FP: {odd_m['fp']}, FN: {odd_m['fn']}")
    ps = scale_array(prem_sc)
    os = scale_array(odd_arr)
    us = np.maximum(ps, os)
    um = cm(al, us)
    print("\nScaled Union Score Metrics:")
    print(f"AUC: {um['auc']:.4f}")
    print(f"Accuracy: {um['accuracy']:.4f}")
    print(f"F1 Score: {um['f1']:.4f}")
    print(f"Precision: {um['precision']:.4f}")
    print(f"Recall: {um['recall']:.4f}")
    print("Confusion Matrix:")
    print(f"TP: {um['tp']}, TN: {um['tn']}")
    print(f"FP: {um['fp']}, FN: {um['fn']}")
    print("\nDONE.")

if __name__ == "__main__":
    md("/kaggle/input/flickr/Flickr.mat")
