import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import (LabelPropagation, LabelSpreading,
                                     SelfTrainingClassifier)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Exibe cada aviso (ex.: ConvergenceWarning) uma vez, em vez de ignorá-los.
warnings.simplefilter("once")

SEED = 73
DATASETS = {
    "wine": {
        "loader": load_wine,
        "title": "Wine (UCI Machine Learning Repository)",
        "url": "https://archive.ics.uci.edu/dataset/109/wine",
    },
    "breast_cancer": {
        "loader": load_breast_cancer,
        "title": "Breast Cancer Wisconsin (Diagnostic) (UCI)",
        "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    },
}


# ----------------------------------------------------------------------------
# Funções auxiliares
# ----------------------------------------------------------------------------
def make_semi_labels(y_train, frac, seed):
    """Revela `frac` dos rótulos por classe (mín. 1). Retorna y_semi (-1 = U) e índices de L."""
    rng = np.random.RandomState(seed)
    y_semi = np.full_like(y_train, -1)
    lab = []
    for c in np.unique(y_train):
        idx = np.where(y_train == c)[0]
        n_c = max(1, int(np.round(frac * len(idx))))
        sel = rng.choice(idx, size=n_c, replace=False)
        y_semi[sel] = c
        lab.extend(sel.tolist())
    return y_semi, np.array(sorted(lab))


def evaluate(y_true, y_pred):
    return {
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "acc": float(accuracy_score(y_true, y_pred)),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }


def lr(seed=SEED):
    return LogisticRegression(max_iter=1000, random_state=seed)


def mean_std(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "runs": [float(v) for v in vals]}


# ----------------------------------------------------------------------------
# Co-Training simplificado (sem uso de rótulos reais de U no treinamento)
# ----------------------------------------------------------------------------
def co_training(X_train, y_train, labeled_idx, unlabeled_idx, view_A, view_B,
                rounds=3, max_per_round=5, conf=0.85, seed=SEED):
    """
    y_train só é consultado para (i) rótulos de L e (ii) AUDITORIA dos
    pseudorrótulos. Os rótulos usados no treino de U são os pseudorrótulos.
    """
    # Conjuntos rotulados separados: LA (treina o modelo A) e LB (treina o modelo B)
    idx_LA = list(labeled_idx); y_LA = list(y_train[labeled_idx])
    idx_LB = list(labeled_idx); y_LB = list(y_train[labeled_idx])
    pool = list(unlabeled_idx)

    total_transferred = 0
    total_audit_errors = 0
    disagreed_samples = set()          # amostras distintas em desacordo
    round_log = []

    for r in range(1, rounds + 1):
        if not pool:
            break

        mA = lr(seed).fit(X_train[idx_LA][:, view_A], y_LA)
        mB = lr(seed).fit(X_train[idx_LB][:, view_B], y_LB)

        Xp = X_train[pool]
        pA = mA.predict_proba(Xp[:, view_A])
        pB = mB.predict_proba(Xp[:, view_B])
        confA, confB = pA.max(axis=1), pB.max(axis=1)
        # converter índice do argmax para o rótulo real da classe
        predA = mA.classes_[pA.argmax(axis=1)]
        predB = mB.classes_[pB.argmax(axis=1)]

        def top_candidates(conf_vec):
            c = np.where(conf_vec >= conf)[0]
            return c[np.argsort(-conf_vec[c])][:max_per_round]

        candA = top_candidates(confA)      # posições no pool
        candB = top_candidates(confB)

        # Divergência: ambos propuseram a mesma amostra com rótulos diferentes
        pos_common = set(candA.tolist()) & set(candB.tolist())
        discard = {p for p in pos_common if predA[p] != predB[p]}
        disagreed_samples.update(pool[p] for p in discard)

        # A ensina B ; B ensina A (somente pseudorrótulos)
        add_to_B = {pool[p]: predA[p] for p in candA if p not in discard}
        add_to_A = {pool[p]: predB[p] for p in candB if p not in discard}

        incorporated = set()
        for idx, lbl in add_to_B.items():
            if idx not in idx_LB:
                idx_LB.append(idx); y_LB.append(lbl)
                incorporated.add(idx)
                total_audit_errors += int(lbl != y_train[idx])   # auditoria apenas
        for idx, lbl in add_to_A.items():
            if idx not in idx_LA:
                idx_LA.append(idx); y_LA.append(lbl)
                incorporated.add(idx)
                total_audit_errors += int(lbl != y_train[idx])   # auditoria apenas

        # Cada exemplo só é incorporado uma vez: sai do pool
        pool = [i for i in pool if i not in incorporated]
        total_transferred += len(incorporated)
        round_log.append({
            "round": r,
            "transferred_round": len(incorporated),
            "to_A": len(add_to_A), "to_B": len(add_to_B),
            "disagreements_round": len(discard),
            "pool_U_remaining": len(pool),
        })

    mA = lr(seed).fit(X_train[idx_LA][:, view_A], y_LA)
    mB = lr(seed).fit(X_train[idx_LB][:, view_B], y_LB)
    return {
        "model_A": mA, "model_B": mB,
        "total_transferred": total_transferred,
        "total_audit_errors": total_audit_errors,
        "unique_disagreements": len(disagreed_samples),
        "rounds": round_log,
        "n_LA": len(idx_LA), "n_LB": len(idx_LB),
    }


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def plot_etapa5(name, X_train, y_train, labeled_idx, y_inferred, classes, target_names, out):
    is_L = np.zeros(len(y_train), dtype=bool)
    is_L[labeled_idx] = True
    emb = {
        "PCA": PCA(n_components=2, random_state=SEED).fit_transform(X_train),
        "t-SNE": TSNE(n_components=2, random_state=SEED,
                      perplexity=min(30, max(5, (len(X_train) - 1) // 4))).fit_transform(X_train),
    }
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for r, (title, Z) in enumerate(emb.items()):
        ax = axes[r, 0]
        ax.scatter(Z[~is_L, 0], Z[~is_L, 1], c="lightgray", s=18, label="U (sem rótulo)")
        for c in classes:
            m = is_L & (y_train == c)
            ax.scatter(Z[m, 0], Z[m, 1], color=cmap(c), s=80, edgecolor="k",
                       marker="*", label=f"L: {target_names[c]}")
        ax.set_title(f"{title} — L vs U"); ax.legend(fontsize=7)

        ax = axes[r, 1]
        for c in classes:
            m = y_inferred == c
            ax.scatter(Z[m, 0], Z[m, 1], color=cmap(c), s=18, label=target_names[c])
        ax.set_title(f"{title} — rótulos inferidos (Label Prop. k=5)"); ax.legend(fontsize=7)

        ax = axes[r, 2]
        for c in classes:
            m = y_train == c
            ax.scatter(Z[m, 0], Z[m, 1], color=cmap(c), s=18, label=target_names[c])
        ax.set_title(f"{title} — rótulos reais (apenas auditoria)"); ax.legend(fontsize=7)
    fig.suptitle(f"{name}: distribuição 2D (somente descritivo)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_etapa7(name, stress, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = {"baseline": "Supervisionado (só L)", "self_training": "Self-Training (τ=0,75)",
              "label_spreading": "Label Spreading (k=5, α=0,2)"}
    for model, lab in labels.items():
        xs = [stress[f]["n_labels_mean"] for f in stress]
        ys = [stress[f][model]["mean"] for f in stress]
        es = [stress[f][model]["std"] for f in stress]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=lab)
    ax.set_xlabel("Nº de rótulos em L (média)")
    ax.set_ylabel("F1-score (macro) no teste")
    ax.set_title(f"{name}: F1 × nº de rótulos (média ± dp, 3 sementes)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Experimento principal
# ----------------------------------------------------------------------------
def run_experiment(name, cfg, out_dir):
    print("\n" + "=" * 60)
    print(f"  EXPERIMENTO: {name.upper()}")
    print("=" * 60)

    data = cfg["loader"]()
    X, y = data.data, data.target
    target_names = [str(t) for t in data.target_names]
    classes, counts = np.unique(y, return_counts=True)
    n_features = X.shape[1]

    # ---------------- ETAPA 1 ----------------
    info = {
        "title": cfg["title"], "url": cfg["url"],
        "n_samples": int(X.shape[0]), "n_features": int(n_features),
        "classes": target_names,
        "class_distribution": dict(zip(target_names, map(int, counts))),
        "missing_values": int(np.isnan(X).sum()),
    }
    print(f"[ETAPA 1] {info}")

    # 20% teste (random_state=73). Estratificado (decisão documentada; o enunciado não exige).
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    scaler = StandardScaler()                       # ajustado SOMENTE no treino
    X_train = scaler.fit_transform(X_tr_raw)
    X_test = scaler.transform(X_te_raw)

    y_semi, labeled_idx = make_semi_labels(y_train, 0.10, SEED)
    unlabeled_idx = np.where(y_semi == -1)[0]
    X_L, y_L = X_train[labeled_idx], y_train[labeled_idx]
    y_U_true = y_train[unlabeled_idx]               # SOMENTE auditoria
    n_U = len(unlabeled_idx)
    print(f"[ETAPA 1] Treino={len(y_train)} (L={len(labeled_idx)}, U={n_U}) | Teste={len(y_test)}")
    for c in classes:
        print(f"          {target_names[c]}: treino={np.sum(y_train == c)}, "
              f"L={np.sum(y_L == c)}, U={np.sum(y_U_true == c)}")

    results = {"dataset": info,
               "step1_split": {"n_train": int(len(y_train)), "n_test": int(len(y_test)),
                               "n_L": int(len(labeled_idx)), "n_U": int(n_U),
                               "f1_average": "macro"}}

    # ---------------- ETAPA 2 ----------------
    t0 = time.time(); m_full = lr().fit(X_train, y_train); t_full = time.time() - t0
    ev_full = evaluate(y_test, m_full.predict(X_test))
    t0 = time.time(); m_few = lr().fit(X_L, y_L); t_few = time.time() - t0
    ev_few = evaluate(y_test, m_few.predict(X_test))
    print(f"[ETAPA 2] Todos os rótulos: F1={ev_full['f1']:.4f} Acc={ev_full['acc']:.4f} t={t_full:.4f}s")
    print(f"[ETAPA 2] Somente L       : F1={ev_few['f1']:.4f} Acc={ev_few['acc']:.4f} t={t_few:.4f}s")
    print(f"[ETAPA 2] Perda de F1 por reduzir rótulos: {ev_full['f1'] - ev_few['f1']:.4f}")
    results["step2_baseline"] = {"full": {**ev_full, "time_s": t_full},
                                 "few": {**ev_few, "time_s": t_few},
                                 "f1_loss": ev_full["f1"] - ev_few["f1"]}

    # ---------------- ETAPA 3: SELF-TRAINING ----------------
    st_results = []
    for tau in (0.60, 0.75, 0.95):
        st = SelfTrainingClassifier(lr(), threshold=tau, max_iter=10)
        t0 = time.time(); st.fit(X_train, y_semi); t_st = time.time() - t0
        ev = evaluate(y_test, st.predict(X_test))
        accepted = (y_semi == -1) & (st.labeled_iter_ > 0)
        n_acc = int(accepted.sum())
        prec = float(accuracy_score(y_train[accepted], st.transduction_[accepted])) if n_acc else None
        st_results.append({"threshold": tau, **ev, "n_accepted": n_acc, "total_u": int(n_U),
                           "coverage": n_acc / n_U, "precision_accepted": prec,
                           "n_iter": int(st.n_iter_), "time_s": t_st})
        prec_txt = f"{prec * 100:.1f}%" if prec is not None else "n/a"
        print(f"[ETAPA 3] τ={tau}: F1={ev['f1']:.4f} Aceitos={n_acc}/{n_U} "
              f"(precisão {prec_txt}) iters={st.n_iter_} t={t_st:.4f}s")
    results["step3_self_training"] = st_results
    st_075 = st_results[1]

    # ---------------- ETAPA 4: CO-TRAINING ----------------
    view_A = [i for i in range(n_features) if i % 2 == 0]
    view_B = [i for i in range(n_features) if i % 2 != 0]
    init = {}
    for v, cols in (("A", view_A), ("B", view_B)):
        m = lr().fit(X_L[:, cols], y_L)
        init[v] = evaluate(y_test, m.predict(X_test[:, cols]))
        print(f"[ETAPA 4] Visão {v} isolada (só L): F1={init[v]['f1']:.4f} Acc={init[v]['acc']:.4f}")

    t0 = time.time()
    co = co_training(X_train, y_train, labeled_idx, unlabeled_idx, view_A, view_B)
    t_co = time.time() - t0
    evA = evaluate(y_test, co["model_A"].predict(X_test[:, view_A]))
    evB = evaluate(y_test, co["model_B"].predict(X_test[:, view_B]))
    best = "A" if evA["f1"] >= evB["f1"] else "B"
    ev_best = evA if best == "A" else evB
    print(f"[ETAPA 4] Co-Training: A F1={evA['f1']:.4f} | B F1={evB['f1']:.4f} | melhor={best} "
          f"(escolhido pelo teste -> viés otimista)")
    print(f"          Transferidos={co['total_transferred']} Desacordos(distintos)="
          f"{co['unique_disagreements']} Erros auditados={co['total_audit_errors']} t={t_co:.4f}s")
    results["step4_co_training"] = {
        "view_A_cols": view_A, "view_B_cols": view_B,
        "init_view_A": init["A"], "init_view_B": init["B"],
        "final_A": evA, "final_B": evB, "best_view": best,
        "total_transferred": co["total_transferred"],
        "unique_disagreements": co["unique_disagreements"],
        "audited_errors": co["total_audit_errors"],
        "n_LA": co["n_LA"], "n_LB": co["n_LB"],
        "time_s": t_co, "rounds": co["rounds"],
    }

    # ---------------- ETAPA 5: LABEL PROPAGATION ----------------
    lp_results = []
    lp_k5 = None
    for k in (3, 5, 15):
        lp = LabelPropagation(kernel="knn", n_neighbors=k)
        t0 = time.time(); lp.fit(X_train, y_semi); t_lp = time.time() - t0
        ev = evaluate(y_test, lp.predict(X_test))
        pred_U = lp.transduction_[unlabeled_idx]
        hits = int(np.sum(pred_U == y_U_true))
        lp_results.append({"k": k, **ev, "hits_u": hits, "total_u": int(n_U),
                           "acc_u": hits / n_U, "time_s": t_lp})
        if k == 5:
            lp_k5 = lp
        print(f"[ETAPA 5] LP k={k}: F1={ev['f1']:.4f} Acertos em U={hits}/{n_U} "
              f"({hits / n_U * 100:.1f}%) t={t_lp:.4f}s")
    results["step5_label_propagation"] = lp_results
    fig5 = os.path.join(out_dir, f"fig_{name}_etapa5_pca_tsne.png")
    plot_etapa5(name, X_train, y_train, labeled_idx, lp_k5.transduction_,
                classes, target_names, fig5)
    print(f"[ETAPA 5] Figura salva em {fig5}")

    # ---------------- ETAPA 6: LABEL SPREADING / RUÍDO ----------------
    def run_graph(y_s):
        out = {}
        lp_ = LabelPropagation(kernel="knn", n_neighbors=5).fit(X_train, y_s)
        out["lp"] = (evaluate(y_test, lp_.predict(X_test)), lp_)
        for a in (0.2, 0.8):
            t0 = time.time()
            ls_ = LabelSpreading(kernel="knn", n_neighbors=5, alpha=a).fit(X_train, y_s)
            t_ = time.time() - t0
            ev_ = evaluate(y_test, ls_.predict(X_test))
            ev_["time_s"] = t_
            out[f"ls_{a}"] = (ev_, ls_)
        return out

    clean = run_graph(y_semi)

    rng = np.random.RandomState(SEED)
    n_corrupt = 2 if len(labeled_idx) >= 6 else 1
    corrupt_idx = rng.choice(labeled_idx, size=n_corrupt, replace=False)
    original = y_semi[corrupt_idx].copy()
    y_noisy = y_semi.copy()
    for i in corrupt_idx:                                  # só L é alterada
        y_noisy[i] = rng.choice([c for c in classes if c != y_semi[i]])
    print(f"[ETAPA 6] Sementes corrompidas {corrupt_idx.tolist()}: "
          f"{original.tolist()} -> {y_noisy[corrupt_idx].tolist()}")
    noisy = run_graph(y_noisy)

    for tag, res in (("limpas", clean), ("ruidosas", noisy)):
        print(f"[ETAPA 6] Sementes {tag}: LP F1={res['lp'][0]['f1']:.4f} | "
              f"LS(α=0,2) F1={res['ls_0.2'][0]['f1']:.4f} | LS(α=0,8) F1={res['ls_0.8'][0]['f1']:.4f}")
    preds_corrupt = {k: noisy[k][1].transduction_[corrupt_idx].tolist() for k in noisy}
    print(f"          Predições finais nas sementes corrompidas {preds_corrupt} "
          f"(rótulo original: {original.tolist()})")
    results["step6_label_spreading"] = {
        "clean": {k: v[0] for k, v in clean.items()},
        "noisy": {k: v[0] for k, v in noisy.items()},
        "corrupted_indices": corrupt_idx.tolist(),
        "original_classes": original.tolist(),
        "noisy_classes": y_noisy[corrupt_idx].tolist(),
        "pred_on_corrupted_seeds": preds_corrupt,
    }

    # ---------------- ETAPA 7: ESTRESSE ----------------
    seeds = (21, 42, 84)
    fractions = (0.05, 0.10, 0.20)
    stress = {}
    for frac in fractions:
        key = f"{int(round(frac * 100))}%"
        acc = {"baseline": [], "self_training": [], "label_spreading": []}
        n_labs = []
        for s in seeds:
            y_s, lab = make_semi_labels(y_train, frac, s)
            n_labs.append(len(lab))
            m = lr(s).fit(X_train[lab], y_train[lab])
            acc["baseline"].append(f1_score(y_test, m.predict(X_test), average="macro"))
            m = SelfTrainingClassifier(lr(s), threshold=0.75, max_iter=10).fit(X_train, y_s)
            acc["self_training"].append(f1_score(y_test, m.predict(X_test), average="macro"))
            m = LabelSpreading(kernel="knn", n_neighbors=5, alpha=0.2).fit(X_train, y_s)
            acc["label_spreading"].append(f1_score(y_test, m.predict(X_test), average="macro"))
        stress[key] = {"n_labels_mean": float(np.mean(n_labs)),
                       **{m: mean_std(v) for m, v in acc.items()}}
        print(f"[ETAPA 7] {key} (≈{np.mean(n_labs):.0f} rótulos): " + " | ".join(
            f"{m}={np.mean(v):.4f}±{np.std(v):.4f}" for m, v in acc.items()))
    fig7 = os.path.join(out_dir, f"fig_{name}_etapa7_curva_f1.png")
    plot_etapa7(name, stress, fig7)
    print(f"[ETAPA 7] Figura salva em {fig7}")

    # Perturbação só em U (ruído gaussiano em 20% de U), 10% de rótulos, 3 sementes.
    # L e teste intactos; nenhum rótulo novo é criado -> simula mudança de distribuição.
    pert = {"self_training": {"clean": [], "perturbed": []},
            "label_spreading": {"clean": [], "perturbed": []},
            "baseline": []}
    for s in seeds:
        y_s, lab = make_semi_labels(y_train, 0.10, s)
        u_idx = np.where(y_s == -1)[0]
        rng = np.random.RandomState(s)
        noisy_u = rng.choice(u_idx, size=int(0.20 * len(u_idx)), replace=False)
        X_pert = X_train.copy()
        X_pert[noisy_u] += rng.normal(0.0, 0.5, size=X_pert[noisy_u].shape)  # σ=0,5 (dados padronizados)

        m = lr(s).fit(X_train[lab], y_train[lab])     # baseline não usa U: não muda
        pert["baseline"].append(f1_score(y_test, m.predict(X_test), average="macro"))
        for tag, Xt in (("clean", X_train), ("perturbed", X_pert)):
            m = SelfTrainingClassifier(lr(s), threshold=0.75, max_iter=10).fit(Xt, y_s)
            pert["self_training"][tag].append(f1_score(y_test, m.predict(X_test), average="macro"))
            m = LabelSpreading(kernel="knn", n_neighbors=5, alpha=0.2).fit(Xt, y_s)
            pert["label_spreading"][tag].append(f1_score(y_test, m.predict(X_test), average="macro"))
    pert_summary = {
        "baseline": mean_std(pert["baseline"]),
        "self_training": {t: mean_std(v) for t, v in pert["self_training"].items()},
        "label_spreading": {t: mean_std(v) for t, v in pert["label_spreading"].items()},
    }
    print("[ETAPA 7] Perturbação em 20% de U (10% de rótulos, 3 sementes):")
    print(f"          Baseline (sem U): {pert_summary['baseline']['mean']:.4f}")
    for m in ("self_training", "label_spreading"):
        print(f"          {m}: limpo={pert_summary[m]['clean']['mean']:.4f} -> "
              f"perturbado={pert_summary[m]['perturbed']['mean']:.4f}")
    results["step7_stress"] = {"curve": stress, "u_perturbation": pert_summary}

    # ---------------- SÍNTESE ----------------
    ls_clean = clean["ls_0.2"][0]
    lp5 = lp_results[1]
    results["consolidated_table"] = [
        {"model": "Supervisionado — somente L", "f1": ev_few["f1"], "acc": ev_few["acc"],
         "time": t_few, "cm": ev_few["cm"], "obs": "10% dos rótulos por classe"},
        {"model": "Supervisionado — todos os rótulos", "f1": ev_full["f1"], "acc": ev_full["acc"],
         "time": t_full, "cm": ev_full["cm"], "obs": "Teto (100% dos rótulos de treino)"},
        {"model": "Self-Training (τ=0,75)", "f1": st_075["f1"], "acc": st_075["acc"],
         "time": st_075["time_s"], "cm": st_075["cm"],
         "obs": f"{st_075['n_accepted']} pseudorrótulos aceitos"},
        {"model": "Co-Training — visão A", "f1": evA["f1"], "acc": evA["acc"],
         "time": t_co, "cm": evA["cm"], "obs": f"{co['total_transferred']} transferidos"},
        {"model": "Co-Training — visão B", "f1": evB["f1"], "acc": evB["acc"],
         "time": t_co, "cm": evB["cm"],
         "obs": f"{co['unique_disagreements']} desacordos, {co['total_audit_errors']} erros auditados"},
        {"model": "Label Propagation (k=5)", "f1": lp5["f1"], "acc": lp5["acc"],
         "time": lp5["time_s"], "cm": lp5["cm"],
         "obs": f"Acertos em U: {lp5['hits_u']}/{lp5['total_u']}"},
        {"model": "Label Spreading (k=5, α=0,2)", "f1": ls_clean["f1"], "acc": ls_clean["acc"],
         "time": ls_clean["time_s"], "cm": ls_clean["cm"], "obs": "Difusão com soft clamping"},
    ]
    print("\n[SÍNTESE] Resultados consolidados no mesmo teste")
    print(f"{'Modelo':40s} {'F1':>7s} {'Acc':>7s} {'Tempo(s)':>9s}")
    for r in results["consolidated_table"]:
        print(f"{r['model']:40s} {r['f1']:7.4f} {r['acc']:7.4f} {r['time']:9.4f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".", help="pasta de saída (JSON e figuras)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_results = {name: run_experiment(name, cfg, args.out) for name, cfg in DATASETS.items()}
    path = os.path.join(args.out, "experimento_pratico_resultados.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nExperimentos finalizados. Resultados em {path}")


if __name__ == "__main__":
    main()