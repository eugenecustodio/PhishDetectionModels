import numpy as np
import pandas as pd
import joblib
import os
import re
from urllib.parse import urlparse

# =====================
# FEATURE EXTRACTION FUNCTION (Same as training)
# =====================
def extract_features(url):
    """Extract all features from a URL"""
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
# INFERENCE PIPELINE
# =====================
def predict_url(url, model_path):
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
        model = joblib.load(model_path)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        return prediction, probability

    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return None, None

# =====================
# MAIN INFERENCE SCRIPT
# =====================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("PHISHING URL DETECTION SYSTEM")
    print("="*50)
    
    # Discover available models
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    available_models = {}
    
    # Categorize models
    print("\nAvailable Models:")
    print("-"*50)
    for i, model_file in enumerate(model_files, 1):
        model_name = model_file.replace('_best_model.joblib', '').replace('.joblib', '')
        category = "Base Model" if "best_model" in model_file else "Ensemble"
        available_models[i] = (model_name, f"models/{model_file}")
        print(f"{i}. {model_name} ({category})")
    
    # Get model selection
    while True:
        try:
            choice = int(input("\nSelect a model (enter number): "))
            if choice in available_models:
                selected_name, selected_path = available_models[choice]
                print(f"\nSelected model: {selected_name}")
                break
            else:
                print("Invalid selection. Please enter a valid number.")
        except ValueError:
            print("Please enter a number.")
    
    # Get URL input
    while True:
        url = input("\nEnter URL to analyze (or 'exit' to quit): ").strip()
        if url.lower() == 'exit':
            break
        if not url:
            print("Please enter a valid URL")
            continue
            
        # Make prediction
        pred, prob = predict_url(url, selected_path)
        
        if pred is not None:
            status = "PHISHING" if pred == 1 else "LEGITIMATE"
            print("\n" + "="*50)
            print(f"URL: {url[:100]}{'...' if len(url)>100 else ''}")
            print(f"Prediction: {status}")
            print(f"Confidence: [Legitimate: {prob[0]:.4f}, Phishing: {prob[1]:.4f}]")
            print("="*50)
        else:
            print("Prediction failed. Please try another URL.")
    
    print("\nExiting phishing detection system. Goodbye!")