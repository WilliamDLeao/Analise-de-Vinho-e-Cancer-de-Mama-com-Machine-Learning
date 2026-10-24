"""
Experimento Prático — Quando os dados não rotulados ajudam?
DECOM-DV | Engenharia de Computação | Tópicos Especiais em Aprendizado de Máquina
Execução completa para Wine e Breast Cancer (Wisconsin Diagnostic).
"""

import sys
import io
import time
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

def run_experiment_for_dataset(name, load_fn, random_state=73):
    print(f"\n=======================================================")
    print(f"       EXECUTANDO EXPERIMENTO PARA: {name.upper()}")
    print(f"=======================================================")
    
    data = load_fn()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = [str(t) for t in data.target_names]
    n_samples, n_features = X.shape
    classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    
    info = {
        "dataset_name": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "classes": target_names,
        "class_distribution": dict(zip(target_names, [int(c) for c in class_counts])),
        "missing_values": int(np.isnan(X).sum())
    }
    print(f"Dataset: {n_samples} amostras, {n_features} atributos, {n_classes} classes: {info['class_distribution']}")
    
    # -------------------------------------------------------------
    # ETAPA 1: DIVISÃO E MASCARAMENTO
    # 20% Teste, 80% Treino (random_state=73)
    # -------------------------------------------------------------
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    
    # Padronização ajustada SOMENTE no treino
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_orig)
    X_test = scaler.transform(X_test_orig)
    
    # Mascaramento: revelar 10% por classe no treino (L), ocultar os demais (U = -1)
    # Garantir pelo menos 1 amostra por classe em L
    np.random.seed(random_state)
    y_semi = np.full_like(y_train, -1)
    labeled_indices = []
    
    for c in classes:
        c_indices = np.where(y_train == c)[0]
        n_c = len(c_indices)
        n_labeled_c = max(1, int(np.round(0.10 * n_c)))
        selected = np.random.choice(c_indices, size=n_labeled_c, replace=False)
        y_semi[selected] = c
        labeled_indices.extend(selected)
        
    labeled_indices = np.array(sorted(labeled_indices))
    unlabeled_indices = np.where(y_semi == -1)[0]
    
    X_L = X_train[labeled_indices]
    y_L = y_train[labeled_indices]
    X_U = X_train[unlabeled_indices]
    y_U_true = y_train[unlabeled_indices]
    
    print(f"Treino total: {len(y_train)} (L: {len(labeled_indices)}, U: {len(unlabeled_indices)}) | Teste: {len(y_test)}")
    for c in classes:
        print(f"  Classe {target_names[c]}: Total Treino={np.sum(y_train==c)}, L={np.sum(y_L==c)}, U={np.sum(y_U_true==c)}")

    # -------------------------------------------------------------
    # ETAPA 2: SUPERVISIONADO (REFERÊNCIAS)
    # -------------------------------------------------------------
    # Teto: 100% supervisionado
    clf_full = LogisticRegression(max_iter=1000, random_state=random_state)
    t0 = time.time()
    clf_full.fit(X_train, y_train)
    t_full = time.time() - t0
    y_pred_full = clf_full.predict(X_test)
    f1_full = f1_score(y_test, y_pred_full, average="macro")
    acc_full = accuracy_score(y_test, y_pred_full)

    # Baseline: Supervisionado apenas com L
    clf_few = LogisticRegression(max_iter=1000, random_state=random_state)
    t0 = time.time()
    clf_few.fit(X_L, y_L)
    t_few = time.time() - t0
    y_pred_few = clf_few.predict(X_test)
    f1_few = f1_score(y_test, y_pred_few, average="macro")
    acc_few = accuracy_score(y_test, y_pred_few)
    cm_few = confusion_matrix(y_test, y_pred_few).tolist()
    
    print(f"\n[ETAPA 2] Supervisionado Full: F1={f1_full:.4f}, Acc={acc_full:.4f}, Tempo={t_full:.4f}s")
    print(f"[ETAPA 2] Baseline (Somente L): F1={f1_few:.4f}, Acc={acc_few:.4f}, Tempo={t_few:.4f}s")

    # -------------------------------------------------------------
    # ETAPA 3: SELF-TRAINING COM DIFERENTES THRESHOLDS
    # -------------------------------------------------------------
    st_results = []
    thresholds = [0.60, 0.75, 0.95]
    for tau in thresholds:
        base_lr = LogisticRegression(max_iter=1000, random_state=random_state)
        st_clf = SelfTrainingClassifier(base_lr, threshold=tau, max_iter=10)
        t0 = time.time()
        st_clf.fit(X_train, y_semi)
        t_st = time.time() - t0
        
        y_pred_test = st_clf.predict(X_test)
        f1_test = f1_score(y_test, y_pred_test, average="macro")
        acc_test = accuracy_score(y_test, y_pred_test)
        
        # Auditoria de pseudorrótulos aceitos
        # Em st_clf.transduction_: rótulos finais inferidos para todo X_train
        # Exemplos incorporados em U: aqueles onde y_semi == -1 e st_clf.labeled_iter_ > 0
        accepted_mask = (y_semi == -1) & (st_clf.labeled_iter_ > 0)
        n_accepted = int(np.sum(accepted_mask))
        n_U = len(unlabeled_indices)
        
        if n_accepted > 0:
            pseudo_labels = st_clf.transduction_[accepted_mask]
            true_labels_accepted = y_train[accepted_mask]
            prec_accepted = float(accuracy_score(true_labels_accepted, pseudo_labels))
        else:
            prec_accepted = 0.0
            
        n_iter = int(st_clf.n_iter_)
        st_results.append({
            "threshold": tau,
            "f1_score": float(f1_test),
            "accuracy": float(acc_test),
            "n_accepted": n_accepted,
            "total_u": n_U,
            "coverage_ratio": float(n_accepted / n_U) if n_U > 0 else 0.0,
            "precision_accepted": prec_accepted,
            "n_iter": n_iter,
            "time_s": float(t_st)
        })
        print(f"[ETAPA 3] Self-Training (tau={tau}): F1={f1_test:.4f}, Aceitos={n_accepted}/{n_U} ({prec_accepted*100:.1f}% precisão), Iters={n_iter}, Tempo={t_st:.4f}s")

    # -------------------------------------------------------------
    # ETAPA 4: CO-TRAINING
    # Divisão em Visão A (índices pares) e Visão B (índices ímpares)
    # -------------------------------------------------------------
    view_A_cols = [i for i in range(n_features) if i % 2 == 0]
    view_B_cols = [i for i in range(n_features) if i % 2 != 0]
    
    # Avaliar visões isoladamente com LogisticRegression no conjunto L
    clf_A_init = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_A_init.fit(X_L[:, view_A_cols], y_L)
    acc_A_init = accuracy_score(y_test, clf_A_init.predict(X_test[:, view_A_cols]))
    f1_A_init = f1_score(y_test, clf_A_init.predict(X_test[:, view_A_cols]), average="macro")
    
    clf_B_init = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_B_init.fit(X_L[:, view_B_cols], y_L)
    acc_B_init = accuracy_score(y_test, clf_B_init.predict(X_test[:, view_B_cols]))
    f1_B_init = f1_score(y_test, clf_B_init.predict(X_test[:, view_B_cols]), average="macro")
    
    print(f"\n[ETAPA 4] Visão A isolada: F1={f1_A_init:.4f}, Acc={acc_A_init:.4f}")
    print(f"[ETAPA 4] Visão B isolada: F1={f1_B_init:.4f}, Acc={acc_B_init:.4f}")
    
    # Execução do Co-Training: 3 rodadas, até 5 amostras com conf >= 0.85
    # L_A e L_B começam com os índices de L
    idx_L_A = list(labeled_indices)
    idx_L_B = list(labeled_indices)
    pool_U = list(unlabeled_indices)
    
    co_rounds = []
    total_transferred = 0
    total_disagreements = 0
    total_audited_errors = 0
    
    t0_co = time.time()
    for r in range(1, 4):
        if len(pool_U) == 0:
            break
            
        model_A = LogisticRegression(max_iter=1000, random_state=random_state)
        model_A.fit(X_train[idx_L_A][:, view_A_cols], y_train[idx_L_A])
        
        model_B = LogisticRegression(max_iter=1000, random_state=random_state)
        model_B.fit(X_train[idx_L_B][:, view_B_cols], y_train[idx_L_B])
        
        # Probabilidades sobre o pool U restante
        probs_A = model_A.predict_proba(X_train[pool_U][:, view_A_cols])
        probs_B = model_B.predict_proba(X_train[pool_U][:, view_B_cols])
        
        max_prob_A = np.max(probs_A, axis=1)
        pred_class_A = np.argmax(probs_A, axis=1)
        
        max_prob_B = np.max(probs_B, axis=1)
        pred_class_B = np.argmax(probs_B, axis=1)
        
        # Selecionar candidatos com conf >= 0.85 (até 5)
        candidates_A_idx_in_pool = np.where(max_prob_A >= 0.85)[0]
        # Ordenar pelos mais confiantes decrescente
        candidates_A_idx_in_pool = candidates_A_idx_in_pool[np.argsort(-max_prob_A[candidates_A_idx_in_pool])][:5]
        
        candidates_B_idx_in_pool = np.where(max_prob_B >= 0.85)[0]
        candidates_B_idx_in_pool = candidates_B_idx_in_pool[np.argsort(-max_prob_B[candidates_B_idx_in_pool])][:5]
        
        cand_A_actual = [pool_U[i] for i in candidates_A_idx_in_pool]
        cand_B_actual = [pool_U[i] for i in candidates_B_idx_in_pool]
        
        # Identificar amostras em comum e verificar se há divergência
        common = set(cand_A_actual).intersection(set(cand_B_actual))
        discard = set()
        for idx in common:
            pos = pool_U.index(idx)
            if pred_class_A[pos] != pred_class_B[pos]:
                discard.add(idx)
                total_disagreements += 1
                
        # Atualização:
        # A ensina B: cand_A adicionado a L_B
        # B ensina A: cand_B adicionado a L_A
        to_add_to_B = []
        for i_in_pool in candidates_A_idx_in_pool:
            actual_idx = pool_U[i_in_pool]
            if actual_idx not in discard:
                lbl = pred_class_A[i_in_pool]
                to_add_to_B.append((actual_idx, lbl))
                if lbl != y_train[actual_idx]:
                    total_audited_errors += 1
                    
        to_add_to_A = []
        for i_in_pool in candidates_B_idx_in_pool:
            actual_idx = pool_U[i_in_pool]
            if actual_idx not in discard:
                lbl = pred_class_B[i_in_pool]
                to_add_to_A.append((actual_idx, lbl))
                if lbl != y_train[actual_idx]:
                    total_audited_errors += 1

        # Aplicar adições
        incorporated_this_round = set()
        for idx, lbl in to_add_to_B:
            if idx not in idx_L_B:
                idx_L_B.append(idx)
                incorporated_this_round.add(idx)
        for idx, lbl in to_add_to_A:
            if idx not in idx_L_A:
                idx_L_A.append(idx)
                incorporated_this_round.add(idx)
                
        # Remover de pool_U
        pool_U = [idx for idx in pool_U if idx not in incorporated_this_round]
        total_transferred += len(incorporated_this_round)
        
        co_rounds.append({
            "round": r,
            "transferred_round": len(incorporated_this_round),
            "pool_U_remaining": len(pool_U),
            "disagreements": len(discard)
        })
        
    t_co = time.time() - t0_co
    
    # Avaliar modelos finais no teste
    final_model_A = LogisticRegression(max_iter=1000, random_state=random_state)
    final_model_A.fit(X_train[idx_L_A][:, view_A_cols], y_train[idx_L_A])
    pred_test_A = final_model_A.predict(X_test[:, view_A_cols])
    f1_co_A = f1_score(y_test, pred_test_A, average="macro")
    acc_co_A = accuracy_score(y_test, pred_test_A)
    
    final_model_B = LogisticRegression(max_iter=1000, random_state=random_state)
    final_model_B.fit(X_train[idx_L_B][:, view_B_cols], y_train[idx_L_B])
    pred_test_B = final_model_B.predict(X_test[:, view_B_cols])
    f1_co_B = f1_score(y_test, pred_test_B, average="macro")
    acc_co_B = accuracy_score(y_test, pred_test_B)
    
    best_co_model = "A" if f1_co_A >= f1_co_B else "B"
    best_f1_co = max(f1_co_A, f1_co_B)
    best_acc_co = acc_co_A if best_co_model == "A" else acc_co_B
    
    print(f"[ETAPA 4] Co-Training Concluído: Modelo A (F1={f1_co_A:.4f}), Modelo B (F1={f1_co_B:.4f}), Melhor={best_co_model}")
    print(f"         Total transferidos={total_transferred}, Desacordos={total_disagreements}, Erros auditados={total_audited_errors}, Tempo={t_co:.4f}s")

    # -------------------------------------------------------------
    # ETAPA 5: LABEL PROPAGATION
    # k = 3, 5, 15
    # -------------------------------------------------------------
    lp_results = []
    k_values = [3, 5, 15]
    for k in k_values:
        lp = LabelPropagation(kernel="knn", n_neighbors=k)
        t0 = time.time()
        lp.fit(X_train, y_semi)
        t_lp = time.time() - t0
        
        y_pred_test = lp.predict(X_test)
        f1_test = f1_score(y_test, y_pred_test, average="macro")
        acc_test = accuracy_score(y_test, y_pred_test)
        
        # Acertos nos rótulos ocultos de U (transdução em U)
        pred_U = lp.transduction_[unlabeled_indices]
        acc_U = accuracy_score(y_U_true, pred_U)
        n_hits_U = int(np.sum(pred_U == y_U_true))
        
        lp_results.append({
            "k": k,
            "f1_score": float(f1_test),
            "accuracy": float(acc_test),
            "hits_u": f"{n_hits_U}/{len(y_U_true)}",
            "acc_u": float(acc_U),
            "time_s": float(t_lp)
        })
        print(f"[ETAPA 5] Label Propagation (k={k}): F1={f1_test:.4f}, Acertos em U={n_hits_U}/{len(y_U_true)} ({acc_U*100:.1f}%), Tempo={t_lp:.4f}s")

    # -------------------------------------------------------------
    # ETAPA 6: LABEL SPREADING E SEMENTES ERRADAS
    # -------------------------------------------------------------
    # Cenário 1: Sementes corretas
    lp_clean = LabelPropagation(kernel="knn", n_neighbors=5)
    lp_clean.fit(X_train, y_semi)
    f1_lp_clean = f1_score(y_test, lp_clean.predict(X_test), average="macro")
    
    ls_clean_02 = LabelSpreading(kernel="knn", n_neighbors=5, alpha=0.2)
    ls_clean_02.fit(X_train, y_semi)
    f1_ls_clean_02 = f1_score(y_test, ls_clean_02.predict(X_test), average="macro")
    acc_ls_clean_02 = accuracy_score(y_test, ls_clean_02.predict(X_test))
    
    ls_clean_08 = LabelSpreading(kernel="knn", n_neighbors=5, alpha=0.8)
    ls_clean_08.fit(X_train, y_semi)
    f1_ls_clean_08 = f1_score(y_test, ls_clean_08.predict(X_test), average="macro")

    # Cenário 2: 1 a 2 sementes deliberadamente alteradas
    y_semi_noisy = np.copy(y_semi)
    # Selecionar 1 ou 2 sementes rotuladas e trocar por classe incorreta
    np.random.seed(random_state)
    n_corrupt = 2 if len(labeled_indices) >= 6 else 1
    corrupt_indices = np.random.choice(labeled_indices, size=n_corrupt, replace=False)
    original_classes = y_semi[corrupt_indices]
    
    for idx in corrupt_indices:
        orig = y_semi[idx]
        other_classes = [c for c in classes if c != orig]
        y_semi_noisy[idx] = np.random.choice(other_classes)
        
    print(f"[ETAPA 6] Sementes corrompidas ({n_corrupt}): Índices {corrupt_indices}, Classes originais {original_classes} -> Trocadas para {y_semi_noisy[corrupt_indices]}")
    
    # Testar com sementes ruidosas
    lp_noisy = LabelPropagation(kernel="knn", n_neighbors=5)
    lp_noisy.fit(X_train, y_semi_noisy)
    f1_lp_noisy = f1_score(y_test, lp_noisy.predict(X_test), average="macro")
    
    ls_noisy_02 = LabelSpreading(kernel="knn", n_neighbors=5, alpha=0.2)
    ls_noisy_02.fit(X_train, y_semi_noisy)
    f1_ls_noisy_02 = f1_score(y_test, ls_noisy_02.predict(X_test), average="macro")
    pred_corrupt_02 = ls_noisy_02.transduction_[corrupt_indices]
    
    ls_noisy_08 = LabelSpreading(kernel="knn", n_neighbors=5, alpha=0.8)
    ls_noisy_08.fit(X_train, y_semi_noisy)
    f1_ls_noisy_08 = f1_score(y_test, ls_noisy_08.predict(X_test), average="macro")
    pred_corrupt_08 = ls_noisy_08.transduction_[corrupt_indices]
    
    print(f"[ETAPA 6] Sementes Limpas: LP F1={f1_lp_clean:.4f} | LS(a=0.2) F1={f1_ls_clean_02:.4f} | LS(a=0.8) F1={f1_ls_clean_08:.4f}")
    print(f"[ETAPA 6] Sementes Ruidosas: LP F1={f1_lp_noisy:.4f} | LS(a=0.2) F1={f1_ls_noisy_02:.4f} | LS(a=0.8) F1={f1_ls_noisy_08:.4f}")
    print(f"         Predições finais nas sementes corrompidas: a=0.2 -> {pred_corrupt_02.tolist()}, a=0.8 -> {pred_corrupt_08.tolist()} (Gabarito real: {original_classes.tolist()})")

    # -------------------------------------------------------------
    # ETAPA 7: DESAFIO DE ESTRESSE
    # (A) Variação da proporção de rótulos: 5%, 10%, 20%
    # (B) Perturbação gaussiana em 20% de U
    # (C) Múltiplas sementes: 21, 42, 84
    # -------------------------------------------------------------
    fractions = [0.05, 0.10, 0.20]
    seeds = [21, 42, 84]
    
    stress_results = {f"{int(frac*100)}%": {"baseline": [], "self_training": [], "label_spreading": []} for frac in fractions}
    
    for frac in fractions:
        for s in seeds:
            np.random.seed(s)
            y_s = np.full_like(y_train, -1)
            lab_idx = []
            for c in classes:
                c_idx = np.where(y_train == c)[0]
                n_lab = max(1, int(np.round(frac * len(c_idx))))
                sel = np.random.choice(c_idx, size=n_lab, replace=False)
                y_s[sel] = c
                lab_idx.extend(sel)
            
            # Baseline
            m_base = LogisticRegression(max_iter=1000, random_state=s)
            m_base.fit(X_train[lab_idx], y_train[lab_idx])
            f1_b = f1_score(y_test, m_base.predict(X_test), average="macro")
            stress_results[f"{int(frac*100)}%"]["baseline"].append(f1_b)
            
            # Self-Training (tau=0.75)
            m_st = SelfTrainingClassifier(LogisticRegression(max_iter=1000, random_state=s), threshold=0.75, max_iter=10)
            m_st.fit(X_train, y_s)
            f1_s = f1_score(y_test, m_st.predict(X_test), average="macro")
            stress_results[f"{int(frac*100)}%"]["self_training"].append(f1_s)
            
            # Label Spreading (a=0.2)
            m_ls = LabelSpreading(kernel="knn", n_neighbors=5, alpha=0.2)
            m_ls.fit(X_train, y_s)
            f1_l = f1_score(y_test, m_ls.predict(X_test), average="macro")
            stress_results[f"{int(frac*100)}%"]["label_spreading"].append(f1_l)

    # Perturbação de 20% em U (ruído gaussiano)
    np.random.seed(random_state)
    n_noisy_u = int(0.20 * len(unlabeled_indices))
    noisy_u_idx = np.random.choice(unlabeled_indices, size=n_noisy_u, replace=False)
    
    X_train_perturbed = np.copy(X_train)
    # Adicionar ruído gaussiano moderado (média 0, std 0.5)
    noise = np.random.normal(loc=0.0, scale=0.5, size=X_train_perturbed[noisy_u_idx].shape)
    X_train_perturbed[noisy_u_idx] += noise
    
    # Avaliar com U perturbado
    st_perturbed = SelfTrainingClassifier(LogisticRegression(max_iter=1000, random_state=random_state), threshold=0.75, max_iter=10)
    st_perturbed.fit(X_train_perturbed, y_semi)
    f1_st_perturbed = f1_score(y_test, st_perturbed.predict(X_test), average="macro")
    
    ls_perturbed = LabelSpreading(kernel="knn", n_neighbors=5, alpha=0.2)
    ls_perturbed.fit(X_train_perturbed, y_semi)
    f1_ls_perturbed = f1_score(y_test, ls_perturbed.predict(X_test), average="macro")
    
    print(f"\n[ETAPA 7] Efeito do Ruído em U (20% das amostras de U corrompidas):")
    print(f"         Self-Training: F1 limpo={st_results[1]['f1_score']:.4f} -> F1 perturbado={f1_st_perturbed:.4f}")
    print(f"         Label Spreading: F1 limpo={f1_ls_clean_02:.4f} -> F1 perturbado={f1_ls_perturbed:.4f}")
    
    # -------------------------------------------------------------
    # SÍNTESE FINAL
    # -------------------------------------------------------------
    summary = {
        "dataset": info,
        "step1_split": {
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_L": len(labeled_indices),
            "n_U": len(unlabeled_indices)
        },
        "step2_baseline": {
            "full_f1": f1_full,
            "full_acc": acc_full,
            "full_time": t_full,
            "few_f1": f1_few,
            "few_acc": acc_few,
            "few_time": t_few,
            "cm_few": cm_few
        },
        "step3_self_training": st_results,
        "step4_co_training": {
            "view_A_init_f1": f1_A_init,
            "view_B_init_f1": f1_B_init,
            "final_f1_A": f1_co_A,
            "final_f1_B": f1_co_B,
            "best_f1": best_f1_co,
            "best_acc": best_acc_co,
            "best_view": best_co_model,
            "total_transferred": total_transferred,
            "disagreements": total_disagreements,
            "audited_errors": total_audited_errors,
            "time_s": t_co,
            "rounds": co_rounds
        },
        "step5_label_propagation": lp_results,
        "step6_label_spreading": {
            "clean": {
                "propagation_f1": f1_lp_clean,
                "spreading_02_f1": f1_ls_clean_02,
                "spreading_08_f1": f1_ls_clean_08
            },
            "noisy": {
                "propagation_f1": f1_lp_noisy,
                "spreading_02_f1": f1_ls_noisy_02,
                "spreading_08_f1": f1_ls_noisy_08,
                "corrupted_indices": corrupt_indices.tolist(),
                "original_classes": original_classes.tolist(),
                "noisy_classes": y_semi_noisy[corrupt_indices].tolist(),
                "pred_corrupt_02": pred_corrupt_02.tolist(),
                "pred_corrupt_08": pred_corrupt_08.tolist()
            }
        },
        "step7_stress": {
            "multi_seed_fractions": {
                frac_str: {
                    model: {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "runs": vals
                    } for model, vals in models.items()
                } for frac_str, models in stress_results.items()
            },
            "u_perturbation": {
                "st_clean_f1": st_results[1]["f1_score"],
                "st_perturbed_f1": f1_st_perturbed,
                "ls_clean_f1": f1_ls_clean_02,
                "ls_perturbed_f1": f1_ls_perturbed
            }
        },
        "consolidated_table": [
            {"model": "Supervisionado - somente L", "f1": f1_few, "acc": acc_few, "time": t_few, "obs": "Apenas 10% dos rótulos por classe (baseline restrito)"},
            {"model": "Supervisionado - todos os rótulos", "f1": f1_full, "acc": acc_full, "time": t_full, "obs": "Teto teórico (oráculo com 100% dos rótulos de treino)"},
            {"model": "Self-Training (tau=0.75)", "f1": st_results[1]["f1_score"], "acc": st_results[1]["accuracy"], "time": st_results[1]["time_s"], "obs": f"{st_results[1]['n_accepted']} pseudorrótulos aceitos ({st_results[1]['precision_accepted']*100:.1f}% precisão)"},
            {"model": f"Co-Training - visão {best_co_model}", "f1": best_f1_co, "acc": best_acc_co, "time": t_co, "obs": f"{total_transferred} amostras transferidas, {total_disagreements} discordâncias"},
            {"model": "Label Propagation (k=5)", "f1": lp_results[1]["f1_score"], "acc": lp_results[1]["accuracy"], "time": lp_results[1]["time_s"], "obs": f"Acertos em U: {lp_results[1]['hits_u']} ({lp_results[1]['acc_u']*100:.1f}%)"},
            {"model": "Label Spreading (k=5, alpha=0.2)", "f1": f1_ls_clean_02, "acc": acc_ls_clean_02, "time": t_lp, "obs": "Difusão com regularização e soft clamping"}
        ]
    }
    
    return summary

def main():
    print("Iniciando bateria completa de experimentos...")
    results_wine = run_experiment_for_dataset("wine", load_wine, random_state=73)
    results_cancer = run_experiment_for_dataset("breast_cancer", load_breast_cancer, random_state=73)
    
    full_output = {
        "wine": results_wine,
        "breast_cancer": results_cancer
    }
    
    with open("experimento_pratico_resultados.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
        
    print("\n\nExperimentos finalizados com sucesso! Resultados salvos em experimento_pratico_resultados.json")

if __name__ == "__main__":
    main()
