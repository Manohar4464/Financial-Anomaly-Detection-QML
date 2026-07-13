import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output, State
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# --- 1. Data Ingestion and Pre-processing ---

def load_and_preprocess_data(filepath, downsample_frac=0.2, random_state=42):
    df = pd.read_csv(filepath)
    df = df.sample(frac=downsample_frac, random_state=random_state).reset_index(drop=True)
    df = df.sort_values(['user_id', 'transaction_time'])
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    
    # New temporal features
    df['hour'] = df['transaction_time'].dt.hour
    df['day_of_week'] = df['transaction_time'].dt.dayofweek
    
    df['time_since_last_transaction'] = df.groupby('user_id')['transaction_time'].diff().dt.total_seconds().fillna(0)
    df['transaction_frequency_for_user'] = df.groupby('user_id')['transaction_time'].transform('count')

    numerical_features = ['amount', 'time_since_last_transaction', 'transaction_frequency_for_user', 'hour', 'day_of_week']
    categorical_features = ['type']

    numeric_transformer = StandardScaler()
    # Explicitly specify categories to include 'suspicious' even if rare
    categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore',
                                            categories=[['deposit', 'purchase', 'suspicious', 'transfer', 'withdrawal']])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X = df[numerical_features + categorical_features]
    X_processed = preprocessor.fit_transform(X)

    return X_processed, df, preprocessor, numerical_features, categorical_features

# --- 2. Train QSVC as binary classifier with class weights ---

def train_qsvc_binary(X_train, y_train, num_features, class_weight=None):
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=quantum_kernel, class_weight=class_weight)
    start_time = time.time()
    qsvc.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"QSVC training took {elapsed:.2f} seconds")
    return qsvc

# --- 3. Classical baseline for comparison ---

def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    return clf

# --- 4. Evaluation and Visualization ---

def evaluate_model(y_true, y_pred, scores):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}, AUC: {auc:.3f}")
    return accuracy, precision, recall, f1, auc, cm

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.title('Confusion Matrix')
    plt.show()

def plot_reconstruction_error_histogram(scores, y_true):
    plt.figure(figsize=(8,5))
    plt.hist(scores[y_true==0], bins=30, alpha=0.6, label='Normal')
    plt.hist(scores[y_true==1], bins=30, alpha=0.6, label='Anomaly')
    plt.xlabel('Anomaly Score (Higher = More Anomalous)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Anomaly Score Distribution')
    plt.show()

# --- 5. Threshold tuning on validation set ---

def find_best_threshold(y_val, scores_val):
    best_f1 = 0
    best_thresh = 0
    for thresh in np.linspace(min(scores_val), max(scores_val), 100):
        y_pred = (scores_val > thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print(f"Best threshold found: {best_thresh:.4f} with F1: {best_f1:.3f}")
    return best_thresh

# --- 6. Dashboard for real-time anomaly prediction ---

def create_dashboard(preprocessor, qsvc, threshold, numerical_features, categorical_features):
    app = Dash(__name__)
    feature_names = numerical_features + categorical_features

    app.layout = html.Div([
        html.H2("Quantum Anomaly Detection Dashboard", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Label(f"{name}:"),
                dcc.Input(id=f"input-{name}", type='text', value='0', style={'marginRight': '10px'})
            ], style={'display': 'inline-block', 'margin': '10px'}) for name in feature_names
        ], style={'textAlign': 'center'}),
        html.Button('Predict Anomaly', id='predict-button', n_clicks=0, style={'margin': '20px'}),
        html.Div(id='prediction-output', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold'})
    ], style={'padding': '20px'})

    @app.callback(
        Output('prediction-output', 'children'),
        Input('predict-button', 'n_clicks'),
        [State(f'input-{name}', 'value') for name in feature_names]
    )
    def predict(n_clicks, *values):
        if n_clicks == 0:
            return ""
        try:
            input_dict = {}
            for name, val in zip(feature_names, values):
                if name in categorical_features:
                    input_dict[name] = val
                else:
                    input_dict[name] = float(val)
            input_df = pd.DataFrame([input_dict])

            # Fill missing features if any
            for feat in ['time_since_last_transaction', 'transaction_frequency_for_user']:
                if feat not in input_df:
                    input_df[feat] = 0 if feat == 'time_since_last_transaction' else 1

            X_proc = preprocessor.transform(input_df)

            score = qsvc.decision_function(X_proc)[0]  # Higher means more normal
            anomaly_score = -score  # Flip sign: higher means more anomalous
            is_anomaly = anomaly_score > threshold
            pred_text = f"Anomaly: {'YES' if is_anomaly else 'NO'} (Anomaly Score: {anomaly_score:.4f})"
            return pred_text
        except Exception as e:
            return f"Error: {str(e)}"

    return app

# --- Main execution ---

if __name__ == "__main__":
    filepath = "transactions_balanced.csv"  # Your dataset path

    # 1. Load and preprocess data with increased downsampling
    X_processed, df_raw, preprocessor, numerical_features, categorical_features = load_and_preprocess_data(filepath, downsample_frac=0.2)

    print("Class distribution after downsampling:")
    print(df_raw['is_anomaly'].value_counts())

    # 2. Split data into train/val/test with stratification
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_processed, df_raw['is_anomaly'].values, test_size=0.2, random_state=42, stratify=df_raw['is_anomaly'].values)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)  # 0.25 x 0.8 = 0.2 val

    # Convert labels to +1 (normal) and -1 (anomaly) for QSVC
    y_train_qsvc = np.where(y_train == 0, 1, -1)
    y_val_qsvc = np.where(y_val == 0, 1, -1)
    y_test_qsvc = np.where(y_test == 0, 1, -1)

    # 3. Train QSVC binary classifier with class weights
    num_features = X_train.shape[1]
    class_weights = {1: 1, -1: max(1, np.sum(y_train_qsvc == 1) / np.sum(y_train_qsvc == -1))}
    qsvc = train_qsvc_binary(X_train, y_train_qsvc, num_features, class_weight=class_weights)

    # 4. Tune threshold on validation set
    val_scores = -qsvc.decision_function(X_val)
    threshold = find_best_threshold(y_val, val_scores)

    # 5. Predict on test set
    test_scores = -qsvc.decision_function(X_test)
    y_pred = (test_scores > threshold).astype(int)

    # 6. Evaluate QSVC model
    print("\nQSVC Model Evaluation on Test Set:")
    accuracy, precision, recall, f1, auc, cm = evaluate_model(y_test, y_pred, test_scores)
    plot_confusion_matrix(cm)
    plot_reconstruction_error_histogram(test_scores, y_test)

    # 7. Train and evaluate classical RandomForest baseline
    rf_clf = train_random_forest(X_train, y_train)
    rf_scores = rf_clf.predict_proba(X_test)[:, 1]  # Probability of anomaly class (1)
    rf_pred = rf_clf.predict(X_test)

    print("\nRandom Forest Baseline Evaluation on Test Set:")
    evaluate_model(y_test, rf_pred, rf_scores)

    # 8. Launch dashboard with QSVC model
    app = create_dashboard(preprocessor, qsvc, threshold, numerical_features, categorical_features)
    app.run(debug=False)