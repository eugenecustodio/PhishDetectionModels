import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence
import xgboost as xgb
import joblib
import os
import warnings
import time
import seaborn as sns
from urllib.parse import urlparse
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directories for outputs
os.makedirs('models', exist_ok=True)
os.makedirs('convergence_plots', exist_ok=True)
os.makedirs('ensemble_plots', exist_ok=True)
os.makedirs('confusion_matrices', exist_ok=True)

# =====================
# FEATURE EXTRACTION FUNCTION
# =====================
def extract_features(url):
    """Extract all 50+ features from a URL"""
    parsed = urlparse(url)
    hostname = parsed.hostname if parsed.hostname else ''
    path = parsed.path if parsed.path else ''
    
    # Calculate entropy safely
    entropy = 0.0
    if len(url) > 0:
        for c in set(url):
            p = url.count(c) / len(url)
            entropy -= p * np.log2(p) if p > 0 else 0
    
    features = {
        "length_url": len(url),
        "length_hostname": len(hostname),
        "ip": 1 if any(char.isdigit() for char in hostname) else 0,
        "nb_dots": url.count('.'),
        "nb_hyphens": url.count('-'),
        "nb_at": url.count('@'),
        "nb_qm": url.count('?'),
        "nb_and": url.count('&'),
        "nb_or": url.count('|'),
        "nb_eq": url.count('='),
        "nb_underscore": url.count('_'),
        "nb_tilde": url.count('~'),
        "nb_percent": url.count('%'),
        "nb_slash": url.count('/'),
        "nb_star": url.count('*'),
        "nb_colon": url.count(':'),
        "nb_comma": url.count(','),
        "nb_semicolumn": url.count(';'),
        "nb_dollar": url.count('$'),
        "nb_space": url.count(' '),
        "nb_www": 1 if "www" in url else 0,
        "nb_com": 1 if ".com" in url else 0,
        "nb_dslash": url.count('//'),
        "http_in_path": 1 if "http" in path else 0,
        "https_token": 1 if "https" in url else 0,
        "ratio_digits_url": sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
        "ratio_digits_host": sum(c.isdigit() for c in hostname) / len(hostname) if hostname else 0,
        "punycode": 1 if re.search(r'xn--', url, re.IGNORECASE) else 0,
        "port": parsed.port if parsed.port else 0,
        "tld_in_path": 1 if any(tld in path for tld in ['.com', '.net', '.org', '.gov', '.edu']) else 0,
        "tld_in_subdomain": 1 if any(tld in hostname for tld in ['.com', '.net', '.org', '.gov', '.edu']) else 0,
        "abnormal_subdomain": 1 if len(hostname.split('.')) > 3 else 0,
        "nb_subdomains": len(hostname.split('.')) - 1,
        "prefix_suffix": 1 if url.startswith("www") else 0,
        "shortening_service": 1 if any(short in url for short in ['bit.ly', 'goo.gl', 'tinyurl.com']) else 0,
        "path_extension": 1 if any(ext in path for ext in ['.exe', '.zip', '.rar', '.tar', '.pdf']) else 0,
        "length_words_raw": len(url.split()),
        "char_repeat": len(set(url)),
        "shortest_words_raw": min(len(word) for word in url.split()) if url.split() else 0,
        "longest_words_raw": max(len(word) for word in url.split()) if url.split() else 0,
        "shortest_word_host": min(len(word) for word in hostname.split('.')) if hostname else 0,
        "longest_word_host": max(len(word) for word in hostname.split('.')) if hostname else 0,
        "shortest_word_path": min(len(word) for word in path.split('/')) if path else 0,
        "longest_word_path": max(len(word) for word in path.split('/')) if path else 0,
        "avg_words_raw": np.mean([len(word) for word in url.split()]) if url.split() else 0,
        "avg_word_host": np.mean([len(word) for word in hostname.split('.')]) if hostname else 0,
        "avg_word_path": np.mean([len(word) for word in path.split('/')]) if path else 0,
        "phish_hints": 1 if any(kw in url.lower() for kw in ['login', 'secure', 'verify', 'account']) else 0,
        "domain_in_brand": 1 if 'apple' in hostname.lower() else 0,
        "brand_in_subdomain": 1 if 'apple' in (hostname.split('.')[0] if hostname else '') else 0,
        "brand_in_path": 1 if 'apple' in path.lower() else 0,
        "suspicious_tld": 1 if hostname.endswith(('.xyz', '.top', '.club', '.gq', '.cf', '.tk')) else 0,
        "entropy": entropy
    }
    return features

# =====================
# DATA PREPROCESSING
# =====================
print("Loading and preprocessing data...")
df = pd.read_csv('url_dataset_se2.csv')
df['status'] = df['status'].map({'phishing': 1, 'legitimate': 0})

y = df['status']
X = df.drop('status', axis=1)

# Save feature names for inference
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.joblib')

# Identify non-numeric columns and label-encode them
non_numeric = X.select_dtypes(include=['object']).columns.tolist()
joblib.dump(non_numeric, 'models/non_numeric_columns.joblib')

if non_numeric:
    print(f"Encoding non-numeric columns: {non_numeric}")
    for col in non_numeric:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        joblib.dump(le, f'models/le_{col}.joblib')  # Save encoders for inference

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.joblib')  # Save scaler for inference

# Reduce precision to save memory
X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)

# =====================
# MODEL DEFINITIONS
# =====================
print("Initializing models and hyperparameter spaces...")

# GPU config - FIXED XGBoost parameters
xgb_gpu_params = {'tree_method': 'gpu_hist', 'gpu_id': 0} if hasattr(xgb, 'gpu_id') else {}

models = {
    "XGBoost": xgb.XGBClassifier(**xgb_gpu_params, eval_metric='logloss', use_label_encoder=False, random_state=42),
    "RandomForest": RandomForestClassifier(n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_jobs=-1, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(n_jobs=-1, solver='saga', max_iter=5000, random_state=42),
    "KNN": KNeighborsClassifier(n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "NeuralNetwork": MLPClassifier(early_stopping=True, random_state=42),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

search_spaces = {
    "XGBoost": {
        'learning_rate': Real(0.005, 0.5, prior='log-uniform'),
        'max_depth': Integer(3, 15),
        'n_estimators': Integer(100, 2000),
        'gamma': Real(0, 10),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.3, 1.0),
        'reg_alpha': Real(0, 20),
        'reg_lambda': Real(0, 20),
        'min_child_weight': Integer(1, 20)
    },
    "RandomForest": {
        'n_estimators': Integer(100, 2000),
        'max_depth': Integer(5, 100),
        'min_samples_split': Integer(2, 50),
        'min_samples_leaf': Integer(1, 30),
        'max_features': Categorical(['sqrt', 'log2', None]),
        'bootstrap': Categorical([True, False])
    },
    "ExtraTrees": {
        'n_estimators': Integer(100, 2000),
        'max_depth': Integer(5, 100),
        'min_samples_split': Integer(2, 50),
        'min_samples_leaf': Integer(1, 30),
        'max_features': Categorical(['sqrt', 'log2', None]),
        'bootstrap': Categorical([True, False])
    },
    "GradientBoosting": {
        'learning_rate': Real(0.005, 0.5, prior='log-uniform'),
        'n_estimators': Integer(100, 2000),
        'max_depth': Integer(3, 15),
        'min_samples_split': Integer(2, 50),
        'min_samples_leaf': Integer(1, 30),
        'subsample': Real(0.5, 1.0),
        'max_features': Categorical(['sqrt', 'log2', None])
    },
    "LogisticRegression": {
        'C': Real(0.001, 1000, prior='log-uniform'),
        'penalty': Categorical(['l1', 'l2', 'elasticnet']),
        'l1_ratio': Real(0, 1),
        'class_weight': Categorical([None, 'balanced'])
    },
    "KNN": {
        'n_neighbors': Integer(3, 100),
        'weights': Categorical(['uniform', 'distance']),
        'p': Integer(1, 5),
        'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'])
    },
    "AdaBoost": {
        'n_estimators': Integer(50, 1000),
        'learning_rate': Real(0.001, 2.0),
        'algorithm': Categorical(['SAMME', 'SAMME.R'])
    },
    "NeuralNetwork": {
        'hidden_layer_sizes': Integer(50, 500),
        'alpha': Real(0.00001, 0.1, prior='log-uniform'),
        'learning_rate_init': Real(0.0001, 0.1),
        'batch_size': Integer(50, 1000),
        'activation': Categorical(['relu', 'tanh', 'logistic']),
        'solver': Categorical(['adam', 'sgd'])
    },
    "NaiveBayes": {},
    "DecisionTree": {
        'max_depth': Integer(3, 100),
        'min_samples_split': Integer(2, 50),
        'min_samples_leaf': Integer(1, 30),
        'criterion': Categorical(['gini', 'entropy', 'log_loss']),
        'splitter': Categorical(['best', 'random'])
    }
}

# =====================
# MODEL TRAINING WITH BAYESIAN OPTIMIZATION
# =====================
results = {}
best_models = {}
optimization_results = {}
cv_scores = {}

print("Starting Bayesian Optimization for base models...")

for name, model in models.items():
    print(f"\n{'='*40}\nTraining {name}\n{'='*40}")
    start_time = time.time()
    
    if name == "NaiveBayes":
        model.fit(X_train_scaled, y_train)
        train_preds = model.predict(X_train_scaled)
        test_preds = model.predict(X_test_scaled)
        
        # Cross-validation for consistency
        cv_acc = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        
        results[name] = {
            'train_accuracy': accuracy_score(y_train, train_preds),
            'test_accuracy': accuracy_score(y_test, test_preds),
            'cv_mean_accuracy': cv_acc.mean(),
            'cv_std_accuracy': cv_acc.std(),
            'f1': f1_score(y_test, test_preds),
            'precision': precision_score(y_test, test_preds),
            'recall': recall_score(y_test, test_preds)
        }
        best_models[name] = model
        joblib.dump(model, f'models/{name}_best_model.joblib', compress=3)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrices/{name}_confusion_matrix.png', dpi=150)
        plt.close()
        
        print(f"NaiveBayes trained in {time.time()-start_time:.2f}s")
        print(f"Test Accuracy: {results[name]['test_accuracy']:.4f}")
        print(f"F1 Score: {results[name]['f1']:.4f}")
        continue

    # Bayesian Optimization
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces[name],
        n_iter=60,
        cv=3,
        n_jobs=-1,
        scoring='neg_log_loss',
        random_state=42,
        return_train_score=True,
        verbose=0
    )
    
    # Fit with progress bar
    with tqdm(total=60, desc=f"Optimizing {name}") as pbar:
        opt.fit(X_train_scaled, y_train, 
                callback=lambda res: pbar.update(1))

    optimization_results[name] = opt
    best_models[name] = opt.best_estimator_
    
    # Get predictions
    train_preds = opt.best_estimator_.predict(X_train_scaled)
    test_preds = opt.best_estimator_.predict(X_test_scaled)
    
    # Cross-validation scores
    cv_acc = cross_val_score(opt.best_estimator_, X_train_scaled, y_train, 
                            cv=5, scoring='accuracy', n_jobs=-1)
    
    results[name] = {
        'best_params': opt.best_params_,
        'train_accuracy': accuracy_score(y_train, train_preds),
        'test_accuracy': accuracy_score(y_test, test_preds),
        'cv_mean_accuracy': cv_acc.mean(),
        'cv_std_accuracy': cv_acc.std(),
        'f1': f1_score(y_test, test_preds),
        'precision': precision_score(y_test, test_preds),
        'recall': recall_score(y_test, test_preds)
    }
    cv_scores[name] = cv_acc
    
    joblib.dump(opt.best_estimator_, f'models/{name}_best_model.joblib', compress=3)

    print(f"Training completed in {time.time()-start_time:.2f}s")
    print(f"Best params: {opt.best_params_}")
    print(f"Test Accuracy: {results[name]['test_accuracy']:.4f}")
    print(f"F1 Score: {results[name]['f1']:.4f}")
    print(f"CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    # Plot convergence
    plt.figure(figsize=(10,6))
    plot_convergence(opt.optimizer_results_[0])
    plt.title(f'{name} Optimization Convergence')
    plt.savefig(f'convergence_plots/{name}_convergence.png', dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrices/{name}_confusion_matrix.png', dpi=150)
    plt.close()

# =====================
# PERFORMANCE VISUALIZATION (BASE MODELS)
# =====================
print("\nGenerating performance graphs for base models...")

# Sort by CV mean accuracy (avoid data leakage)
sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_mean_accuracy'], reverse=True)
model_names = [x[0] for x in sorted_results]
test_accs = [x[1]['test_accuracy'] for x in sorted_results]
cv_means = [x[1]['cv_mean_accuracy'] for x in sorted_results]
cv_stds = [x[1]['cv_std_accuracy'] for x in sorted_results]

# Create comparison plot
plt.figure(figsize=(16,10))
x_pos = np.arange(len(model_names))

# Bar width
width = 0.35

# Plot test accuracy
rects1 = plt.bar(x_pos - width/2, test_accs, width, 
                 color='skyblue', label='Test Accuracy')

# Plot CV accuracy with error bars
rects2 = plt.bar(x_pos + width/2, cv_means, width, 
                 color='salmon', yerr=cv_stds, 
                 ecolor='black', capsize=5, label='CV Accuracy')

plt.axhline(y=max(test_accs), color='r', linestyle='--', alpha=0.7, linewidth=2)
plt.title('Base Model Performance Comparison', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(x_pos, model_names, rotation=45, ha='right', fontsize=12)
plt.ylim(min(min(test_accs), min(cv_means))-0.05, max(max(test_accs), max(cv_means))+0.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='lower right')

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('base_model_performance.png', dpi=150)
plt.close()

# =====================
# ENSEMBLE CONSTRUCTION
# =====================
print("\nBuilding ensemble models with Logistic Regression meta-learner...")
ensemble_results = {}
ensemble_objects = {}

# Sort by CV mean accuracy (avoid test set leakage)
sorted_models = sorted(best_models.items(), 
                      key=lambda x: results[x[0]]['cv_mean_accuracy'], 
                      reverse=True)

# Ensemble sizes
num_models = len(best_models)
ensemble_sizes = [k for k in [3, 5, 7] if k <= num_models]

for k in ensemble_sizes:
    selected = sorted_models[:k]
    ens_name = f"Top_{k}_Ensemble"
    print(f"\nCreating {ens_name} with models (sorted by CV accuracy):")
    for i, (model_name, _) in enumerate(selected, 1):
        print(f"{i}. {model_name} (CV Acc: {results[model_name]['cv_mean_accuracy']:.4f})")

    # Use best parameters but new instances to avoid refitting issues
    base_models = [(name, type(model)(**model.get_params())) for name, model in selected]

    
    ens = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(n_jobs=-1, solver='saga', 
                                         max_iter=5000, class_weight='balanced',
                                         random_state=42),
        n_jobs=-1,
        cv=5  # Use cross-validation for meta-features
    )

    print(f"Training {ens_name}...")
    start_time = time.time()
    ens.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    test_preds = ens.predict(X_test_scaled)
    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    
    ensemble_results[ens_name] = {
        'models': [n for n, _ in selected],
        'test_accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'train_time': train_time
    }
    ensemble_objects[ens_name] = ens
    joblib.dump(ens, f'models/{ens_name}.joblib', compress=3)

    print(f"Ensemble training time: {train_time:.2f}s")
    print(f"Ensemble accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'{ens_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrices/{ens_name}_confusion_matrix.png', dpi=150)
    plt.close()

# =====================
# ENSEMBLE PERFORMANCE VISUALIZATION
# =====================
print("\nGenerating performance graphs for ensembles...")
sorted_ens = sorted(ensemble_results.items(), 
                   key=lambda x: x[1]['test_accuracy'], 
                   reverse=True)
names_ens = [x[0] for x in sorted_ens]
accs_ens = [x[1]['test_accuracy'] for x in sorted_ens]
f1_scores = [x[1]['f1'] for x in sorted_ens]

# Create subplots
fig, ax1 = plt.subplots(figsize=(14, 8))

# Bar positions
x_pos = np.arange(len(names_ens))

# Plot accuracy
bars = ax1.bar(x_pos - 0.2, accs_ens, 0.4, color='lightgreen', label='Accuracy')
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.set_ylim(min(accs_ens)-0.05, max(accs_ens)+0.05)

# Create second axis for F1 score
ax2 = ax1.twinx()
line = ax2.plot(x_pos + 0.2, f1_scores, 'o-', color='darkred', 
               linewidth=2, markersize=8, label='F1 Score')
ax2.set_ylabel('F1 Score', fontsize=14)
ax2.set_ylim(min(f1_scores)-0.05, max(f1_scores)+0.05)

# Add model names
model_counts = [f"{k} models\n({', '.join(ensemble_results[name]['models'][:3])}...)"
                for name in names_ens]
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_counts, rotation=15, ha='right', fontsize=11)

# Add titles and legends
ax1.set_title('Ensemble Model Performance', fontsize=16)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower right')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('ensemble_performance.png', dpi=150)
plt.close()

# =====================
# FINAL REPORT
# =====================
print("Generating final report...")
with open('model_performance_report.txt','w') as f:
    f.write("PHISHING DETECTION MODEL PERFORMANCE REPORT\n")
    f.write("="*80 + "\n")
    f.write(f"Dataset: {df.shape[0]} samples, {df.shape[1]-1} features\n")
    f.write(f"Train size: {X_train_scaled.shape[0]}, Test size: {X_test_scaled.shape[0]}\n")
    f.write("="*80 + "\n\n")

    f.write("BASE MODELS (Sorted by CV Accuracy):\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Model':<20}{'Test Acc':>10}{'CV Acc':>10}{'F1':>10}{'Precision':>10}{'Recall':>10}\n")
    f.write("-"*80 + "\n")
    for name, res in sorted_results:
        f.write(f"{name:<20}{res['test_accuracy']:>10.4f}{res['cv_mean_accuracy']:>10.4f}"
                f"{res['f1']:>10.4f}{res['precision']:>10.4f}{res['recall']:>10.4f}\n")
    f.write("\n")

    f.write("\nENSEMBLE MODELS (Sorted by Test Accuracy):\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Ensemble':<20}{'Accuracy':>10}{'F1':>10}{'Precision':>10}{'Recall':>10}{'Train Time (s)':>15}\n")
    f.write("-"*80 + "\n")
    for name, res in sorted_ens:
        f.write(f"{name:<20}{res['test_accuracy']:>10.4f}{res['f1']:>10.4f}"
                f"{res['precision']:>10.4f}{res['recall']:>10.4f}{res['train_time']:>15.2f}\n")
    f.write("\n")

    best_base = sorted_results[0][0]
    best_ensemble = sorted_ens[0][0] if sorted_ens else "N/A"
    
    f.write("\n" + "="*80 + "\n")
    f.write(f"BEST BASE MODEL: {best_base} (Test Acc: {results[best_base]['test_accuracy']:.4f})\n")
    if sorted_ens:
        f.write(f"BEST ENSEMBLE: {best_ensemble} (Test Acc: {ensemble_results[best_ensemble]['test_accuracy']:.4f})\n")
    f.write("="*80 + "\n")

print("\nTraining complete! All artifacts saved:")
print(f"- Trained models: models/ directory")
print(f"- Convergence plots: convergence_plots/ directory")
print(f"- Confusion matrices: confusion_matrices/ directory")
print(f"- Performance plots: base_model_performance.png, ensemble_performance.png")
print(f"- Full report: model_performance_report.txt")

# =====================
# INFERENCE PIPELINE
# =====================
def predict_url(url, model_name="XGBoost"):
    """End-to-end prediction pipeline for a single URL"""
    try:
        # 1. Load artifacts
        feature_names = joblib.load('models/feature_names.joblib')
        non_numeric_columns = joblib.load('models/non_numeric_columns.joblib')
        scaler = joblib.load('models/scaler.joblib')
        
        # 2. Load encoders
        label_encoders = {}
        for col in non_numeric_columns:
            le_path = f'models/le_{col}.joblib'
            if os.path.exists(le_path):
                label_encoders[col] = joblib.load(le_path)
        
        # 3. Extract features
        features = extract_features(url)
        
        # 4. Create input DataFrame
        input_data = pd.DataFrame([features], columns=feature_names)
        
        # 5. Apply preprocessing
        for col in non_numeric_columns:
            if col in input_data.columns and col in label_encoders:
                # Handle unseen labels
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
                except ValueError:
                    # Assign unknown class
                    input_data[col] = len(label_encoders[col].classes_)
        
        input_scaled = scaler.transform(input_data)
        
        # 6. Load model and predict
        model_path = f'models/{model_name}_best_model.joblib' if model_name != "Top" else f'models/{model_name}.joblib'
        if not os.path.exists(model_path):
            # Try ensemble model
            model_path = f'models/{model_name}.joblib'
        
        model = joblib.load(model_path)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        return prediction, probability
    
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return None, None

# =====================
# DEMONSTRATE INFERENCE
# =====================
print("\n\nDEMONSTRATING INFERENCE:")
print("=======================")

# Example URLs
test_urls = [
    "https://www.apple.com/shop",  # Legitimate
    "https://login-facebook-secure.xyz/login.php?user=test",  # Phishing
    "https://bit.ly/suspicious-download"  # Phishing
]

for url in test_urls:
    pred, prob = predict_url(url, model_name="Top_5_Ensemble")
    status = "PHISHING" if pred == 1 else "LEGITIMATE"
    print(f"\nURL: {url[:60]}...")
    print(f"Prediction: {status}")
    print(f"Confidence: [Legitimate: {prob[0]:.4f}, Phishing: {prob[1]:.4f}]")

print("\nInference pipeline is ready! Use predict_url() function for predictions.")