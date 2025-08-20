# Imports
import os
import ssl
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, os
import matplotlib.cm as cm
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

import tensorflow as tf


os.makedirs('plots', exist_ok=True) # Create plots dir
plt.ioff() 


# Custom Mahalanobis Classifier
class MahalanobisClassifier():
    def __init__(self, samples, labels, m=None, rho_mode="mean"):
        self.clusters = {}
        for lbl in np.unique(labels):
            self.clusters[lbl] = samples[labels == lbl]

        self.m = m  # threshold m
        self.rho_mode = rho_mode  # mean or max for rho
        self.mean_vec = {}
        self.cov_matrices = {}
        self.eigenvalues = {}
        self.eigenvec = {}
        self.rho = {}
        epsilon = 1e-6  # For num. stability adding small value for ev decomp

        # calc mean and covariance matrices for each class
        for lbl, data in self.clusters.items():
            self.mean_vec[lbl] = np.mean(data, axis=0)
            cov_matrix = np.cov(data, rowvar=False) + epsilon * np.eye(data.shape[1])
            self.cov_matrices[lbl] = cov_matrix

            # eigen-decomp
            eigenvalues, eigenvec = np.linalg.eigh(cov_matrix)
            x = np.argsort(eigenvalues)[::-1]
            sorted_eigenvalues = eigenvalues[x]
            sorted_eigenvec = eigenvec[:, x]

            self.eigenvalues[lbl] = sorted_eigenvalues
            self.eigenvec[lbl] = sorted_eigenvec

            # calc rho based on residual eigenvalues
            if self.m is not None and self.m < len(sorted_eigenvalues):
                residual_eigenvalues = sorted_eigenvalues[self.m:]
                rho = np.mean(residual_eigenvalues) if self.rho_mode == "mean" else np.max(residual_eigenvalues)
                self.rho[lbl] = rho
            else:
                self.rho[lbl] = None

    # mahalanobis with Probabilistic Subspace Learning
    def mahalanobis_psl(self, x, lbl):
        mean = self.mean_vec[lbl]
        proj = np.dot(x - mean, self.eigenvec[lbl])

        eigenvalues_principal = self.eigenvalues[lbl][:self.m]
        dist1 = -0.5 * np.sum((proj[:self.m] ** 2) / eigenvalues_principal)

        dist2 = 0
        if self.rho[lbl] is not None:
            rho = self.rho[lbl]
            dist2 = -0.5 * np.sum((proj[self.m:] ** 2) / rho)

        b_i = 0
        total_dist = dist1 + dist2 + b_i
        return total_dist

    def predict_probab(self, unlabeled_samples):
        dists = []
        for lbl in self.clusters:
            tmp_dists = np.array([
                self.mahalanobis_psl(sample, lbl)
                for sample in unlabeled_samples
            ])
            dists.append(tmp_dists)
        dists = np.column_stack(dists)

        max_dists = np.max(dists, axis=1, keepdims=True)
        exp_dists = np.exp(dists - max_dists)
        probabilities = exp_dists / np.sum(exp_dists, axis=1, keepdims=True)
        return probabilities

    def predict_class(self, unlabeled_samples):
        probas = self.predict_probab(unlabeled_samples)
        return np.argmax(probas, axis=1)



# Metrics
def calc_metrics(cm, classifier="Classifier", n_comp=0, return_values=False):
    """
    returns (acc, ppv, tpr, fpr, f1) from a cm avged over classes
    """
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)

    acc = cm.trace() / cm.sum()
    prec = TP / (TP + FP + 1e-12)
    tpr  = TP / (TP + FN + 1e-12)
    fpr  = FP / (FP + TN + 1e-12)
    f1   = 2 * (prec * tpr) / (prec + tpr + 1e-12)

    if return_values:
        return float(acc), float(np.nanmean(prec)), float(np.nanmean(tpr)), \
               float(np.nanmean(fpr)), float(np.nanmean(f1))
   
    return None


#exp 1 metrics table
def exp1_table(rows, out_path="plots/experiment1_metrics.png"):
    fig, ax = plt.subplots(figsize=(8.5, 1.8))
    ax.axis('off')
    headers = ["Method", "Components", "Overall Acc.", "PPV", "TPR", "FPR", "F1"]

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc='center', loc='center', edges='closed')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.2)

    title = rf'Additional Metrics for PCA and LDA for 9 and 50 Components'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=8)


    # header
    for j in range(len(headers)):
        table[(0, j)].set_text_props(weight='bold')

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved exp1 table: {out_path}")


def experiment1(X_train, X_test, y_train, y_test, exp2=False):
    """Experiment 1: Comparing PCA and LDA Classification Accuracy by Components"""
    pca_acc, lda_acc = [], []
    components_pca = [1, 5, 9, 20, 50]
    components_lda = [1, 2, 5, 9]

    exp1_rows = []  # [Method, Components, Acc, PPV, TPR, FPR, F1]

    # PCA
    for n in tqdm(components_pca, desc="PCA Components"):
        pca = PCA(n_components=n, random_state=0)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca  = pca.transform(X_test)

        clf = MahalanobisClassifier(X_train_pca, y_train, m=n)
        y_pred = clf.predict_class(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        pca_acc.append(acc)

        cm = confusion_matrix(y_test, y_pred)
        acc_m, ppv, tpr, fpr, f1 = calc_metrics(cm, classifier=f"PCA (n_components={n})",
                                                n_comp=n, return_values=True)

        if n in (9, 50):  #rows for n=9, n=50
            exp1_rows.append(["PCA", str(n),
                              f"{acc_m:.3f}", f"{ppv:.3f}", f"{tpr:.3f}",
                              f"{fpr:.3f}", f"{f1:.3f}"])

    # LDA
    for n in components_lda:
        lda = LDA(n_components=n)
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda  = lda.transform(X_test)

        clf = MahalanobisClassifier(X_train_lda, y_train, m=n)
        y_pred = clf.predict_class(X_test_lda)
        acc = accuracy_score(y_test, y_pred)
        lda_acc.append(acc)

        cm = confusion_matrix(y_test, y_pred)
        acc_m, ppv, tpr, fpr, f1 = calc_metrics(cm, classifier=f"LDA (n_components={n})",
                                                n_comp=n, return_values=True)

        if n == 9:
            exp1_rows.append(["LDA", str(n),
                              f"{acc_m:.3f}", f"{ppv:.3f}", f"{tpr:.3f}",
                              f"{fpr:.3f}", f"{f1:.3f}"])

    # Plotting
    if not exp2:
        plt.figure(figsize=(12, 8))
        plt.plot(components_pca, pca_acc, marker='o', label='PCA')
        plt.plot(components_lda, lda_acc, marker='s', label='LDA')
        plt.title("Classification Accuracy depending on Number of Components for PCA and LDA")
        plt.xlabel('Number of Components')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        all_components = sorted(set(components_lda + components_pca))
        plt.xticks(all_components)
        plt.tight_layout()
        plt.savefig('plots/experiment1_pca_vs_lda_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Plot saved: plots/experiment1_pca_vs_lda_comparison.png")

    
    order = [("PCA", "9"), ("PCA", "50"), ("LDA", "9")]
    exp1_rows_sorted = [next(row for row in exp1_rows if row[0] == m and row[1] == c) for m, c in order]
    exp1_table(exp1_rows_sorted)

    return pca_acc, lda_acc, components_pca, components_lda


#Experiment 2
def experiment2(X_train, X_test, y_train, y_test, components_lda, lda_acc):
    """Experiment 2: Classification Accuracy Analysis for PCA+LDA by Components"""
    comp_pl_pca = [10, 20, 50, 100, 200, 400, 600]
    comp_pl_lda = [1, 2, 5, 9]

    pca_lda_acc = {pca: {l: [] for l in comp_pl_lda} for pca in comp_pl_pca}

    for n in tqdm(comp_pl_pca, desc="PCA + LDA PCA Components"):
        pca = PCA(n_components=n, random_state=0)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        for l in comp_pl_lda:
            lda_components = min(l, X_train_pca.shape[1])
            lda = LDA(n_components=lda_components)
            X_train_pca_lda = lda.fit_transform(X_train_pca, y_train)
            X_test_pca_lda = lda.transform(X_test_pca)

            clf = MahalanobisClassifier(X_train_pca_lda, y_train, m=lda_components)
            y_pred = clf.predict_class(X_test_pca_lda)
            accuracy_pca_lda = accuracy_score(y_test, y_pred)
            pca_lda_acc[n][l].append(accuracy_pca_lda)

            cm_pca_lda = confusion_matrix(y_test, y_pred)
            calc_metrics(cm_pca_lda, classifier=f"PCA + LDA (PCA n={n}, LDA n={l})", n_comp=l)



    # Color map
    all_n_values = comp_pl_pca
    colors = cm.rainbow(np.linspace(0, 1, len(all_n_values)))
    n_color_map = {n: color for n, color in zip(all_n_values, colors)}
    n_color_map[0] = 'red'
    max_n_value = max(all_n_values)
    n_color_map[max_n_value] = 'grey'

    for idx, lda_comp in enumerate(components_lda):
        accuracies = []
        labels = []
        lda_index = components_lda.index(lda_comp)
        lda_accuracy = lda_acc[lda_index]
        accuracies.append(lda_accuracy)
        labels.append(0)

        for pca_comp in comp_pl_pca:
            accuracy = pca_lda_acc[pca_comp][lda_comp][0]
            accuracies.append(accuracy)
            labels.append(pca_comp)

        plt.figure(figsize=(8, 6))
        plt.boxplot([accuracies], showmeans=False, medianprops=dict(color='black'))

        for j, (accuracy, label) in enumerate(zip(accuracies, labels)):
            color = 'red' if label == 0 else ('grey' if label == max_n_value else n_color_map[label])
            x_position = 1 + (j - len(accuracies) / 2) * 0.05
            plt.scatter(x_position, accuracy, color=color)

        plt.axhline(y=lda_accuracy, color='red', linestyle='-', linewidth=1)
        plt.scatter([], [], color='red', label='n=0 (LDA only)')
        for n, color in n_color_map.items():
            if n != 0:
                plt.scatter([], [], color=color, label=f'n={n}')

        plt.title(f"Classification Accuracies for\nPCA + LDA with {lda_comp} LDA Component{'s' if lda_comp > 1 else ''}")
        plt.legend(title="Number of PCA Components")
        plt.xlabel(f'LDA Components = {lda_comp}')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'plots/experiment2_pca_lda_combination_lda{lda_comp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: plots/experiment2_pca_lda_combination_lda{lda_comp}.png")

    return pca_lda_acc


#tables for experiment 3
def exp3_table(df, rho_mode):

    filename = f"experiment3_{rho_mode}_rho.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')

    header = ['PCA Components'] + [str(col) for col in df.columns]
    rows = []
    for idx, row in df.iterrows():
        rows.append([str(idx)] + [f'{val:.3f}' if pd.notna(val) else 'N/A' for val in row])

    table = ax.table(cellText=rows,
                     colLabels=header,
                     cellLoc='center',
                     loc='center',
                     edges='closed')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.35)

    for j in range(len(header)):
        table[(0, j)].set_text_props(weight='bold')
    for i in range(1, len(rows)+1):
        table[(i, 0)].set_text_props(weight='bold')

    title = rf'Classification Accuracy depending on PCA Components and Threshold $m$ for $\rho_{{{rho_mode}}}$'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=8)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    left_bbox  = table[(0, 1)].get_window_extent(renderer=renderer)
    right_bbox = table[(0, len(header)-1)].get_window_extent(renderer=renderer)

    # coords for rect above header
    x0, y0 = fig.transFigure.inverted().transform((left_bbox.x0, left_bbox.y1))
    # match height
    header_h_px = left_bbox.y1 - left_bbox.y0
    x1, y1 = fig.transFigure.inverted().transform((right_bbox.x1, right_bbox.y1 + header_h_px * 0.85))

    rect = plt.Rectangle((x0, y0), x1-x0, y1-y0,
                         fill=False, color='black', lw=1.0,
                         transform=fig.transFigure, clip_on=False)
    fig.patches.append(rect)

    fig.text((x0+x1)/2, (y0+y1)/2, r'Threshold $m$',
             ha='center', va='center', fontsize=12, fontweight='bold')

    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Table saved as image: plots/{filename}")


def experiment3(X_train, X_test, y_train, y_test):
    """Experiment 3: Classification Accuracy Analysis for PCA Probabilistic Subspace Learning by Components and Threshold m"""
    max_components = min(X_train.shape[0], X_train.shape[1])

    n_comp_full = [10, 20, 50, 100, 200]
    n_comp = [n for n in n_comp_full if n <= max_components]
    if not n_comp or len(n_comp) < 3:
        n_comp = [min(5, max_components), min(10, max_components), min(15, max_components)]
        n_comp = [n for n in n_comp if n > 0 and n <= max_components]

    print(f"Using PCA components: {n_comp} (max possible: {max_components})")

    m_perc = [60, 80, 90, 95, 100]

    mean_acc = {n: {} for n in n_comp}
    max_acc  = {n: {} for n in n_comp}

    for rho in ["mean", "max"]:
        results = mean_acc if rho == "mean" else max_acc

        for n in tqdm(n_comp, desc=f"Rho Mode: {rho}"):
            for m_percent in m_perc:
                m = int(n * (m_percent / 100))
                if m == 0 or m > n:
                    results[n][f"{m_percent}%"] = None
                    continue

                pca = PCA(n_components=n)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca  = pca.transform(X_test)

                classifier = MahalanobisClassifier(X_train_pca, y_train, m=m, rho_mode=rho)
                y_pred = classifier.predict_class(X_test_pca)
                accuracy = accuracy_score(y_test, y_pred)
                results[n][f"{m_percent}%"] = accuracy

    mean_df = pd.DataFrame(mean_acc).T.sort_index()
    max_df  = pd.DataFrame(max_acc).T.sort_index()

    exp3_table(mean_df, "avg")
    exp3_table(max_df, "up")

    print("\nTable saved:")
    print("- plots/experiment3_avg_rho_table.png")
    print("- plots/experiment3_up_rho_table.png")

    return mean_df, max_df


# Data loading
def load_data():
    """Load and preprocess MNIST dataset."""
    print("Loading and preprocessing MNIST dataset...")
    (X_train_orig, y_train_orig), (X_test_orig, y_test_orig) = tf.keras.datasets.mnist.load_data()

    X_train_flat = X_train_orig.reshape(X_train_orig.shape[0], -1)
    X_test_flat  = X_test_orig.reshape(X_test_orig.shape[0], -1)
    X = np.vstack([X_train_flat, X_test_flat])
    y = np.hstack([y_train_orig, y_test_orig])
    print("Loaded MNIST dataset")

    X = X.astype(np.float64) / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    return X_train, X_test, y_train, y_test



# main
def main():
    X_train, X_test, y_train, y_test = load_data()

    print("\n" + "-"*120)
    print("Experiments:")
    print("-"*120)
    print("1. Experiment 1: Comparing PCA and LDA Classification Accuracy by Components")
    print("2. Experiment 2: Classification Accuracy Analysis for PCA+LDA by Components")
    print("3. Experiment 3: Classification Accuracy Analysis for PCA Probabilistic Subspace Learning by Components and Threshold m")
    print("4. Run All Experiments")
    print("-"*120)

    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()

            if choice == '1':
                print("Running Experiment 1: Comparing PCA and LDA Classification Accuracy by Components")
                pca_acc, lda_acc, components_pca, components_lda = experiment1(X_train, X_test, y_train, y_test)
            elif choice == '2':
                print("\nRunning Experiment 2: Classification Accuracy Analysis for PCA+LDA by Components")
                pca_acc, lda_acc, components_pca, components_lda = experiment1(X_train, X_test, y_train, y_test, exp2=True)
                experiment2(X_train, X_test, y_train, y_test, components_lda, lda_acc)
            elif choice == '3':
                print("\nRunning Experiment 3: Classification Accuracy Analysis for PCA Probabilistic Subspace Learning by Components and Threshold m")
                experiment3(X_train, X_test, y_train, y_test)
            elif choice == '4':
                print("Running all experiments...")
                print("Running Experiment 1: PCA vs LDA Comparison")
                pca_acc, lda_acc, components_pca, components_lda = experiment1(X_train, X_test, y_train, y_test)
                print("\nRunning Experiment 2: Classification Accuracy Analysis for PCA+LDA by Components")
                experiment2(X_train, X_test, y_train, y_test, components_lda, lda_acc)
                print("\nRunning Experiment 3: Classification Accuracy Analysis for PCA Probabilistic Subspace Learning by Components and Threshold m")
                experiment3(X_train, X_test, y_train, y_test)
            else:
                print("choose 1-4.")
                continue

            break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue


if __name__ == "__main__":
    main()
