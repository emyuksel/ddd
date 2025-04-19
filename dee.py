import numpy as np
import scipy.io
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# SMAPE hesaplama fonksiyonu
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Veriyi Yükleme
data = scipy.io.loadmat('Gas_Turbine_Co_NoX_2015.mat')
features = data['feat']  # Özellikler
labels = data['lbl2'][:, 0]  # Etiketler

# Veri Kümelerinin Özeti
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)
print("Feature Statistics:")
for i in range(features.shape[1]):
    print(f"Feature {i+1}: Min={features[:,i].min()}, Max={features[:,i].max()}, Mean={features[:,i].mean():.2f}, Std={features[:,i].std():.2f}")
print(f"lbl2: Min={labels.min()}, Max={labels.max()}, Mean={labels.mean():.2f}, Std={labels.std():.2f}")

# Pandas DataFrame'e Çevirme
df_features = pd.DataFrame(features, columns=[f'Feature{i+1}' for i in range(features.shape[1])])
df_labels = pd.DataFrame(labels, columns=['lbl2'])

# Özelliklerin Dağılımı
plt.figure(figsize=(16, 8))
for i, column in enumerate(df_features.columns):
    plt.scatter(range(len(df_features)), df_features[column], label=column, alpha=0.6)
plt.title("Feature Scatter Plot")
plt.xlabel("Index")
plt.ylabel("Feature Values")
plt.legend()
plt.show()

# lbl2'nin Dağılımı
plt.figure(figsize=(8, 6))
sns.histplot(labels, bins=30, kde=True)
plt.title("lbl2 Distribution")
plt.xlabel("lbl2")
plt.ylabel("Frequency")
plt.show()

# Korelasyon Matrisi
correlation_matrix = df_features.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# 2. K-Fold Cross Validation
kf = KFold(n_splits=2, shuffle=True, random_state=42)

# Model 1: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Model 2: Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Model Performansının Hesaplanması
models = {
    "Random Forest": rf_model,
    "Gradient Boosting": gbr_model
}

results = {}
all_predictions = {}
for name, model in models.items():
    mae_scores = []
    smape_scores = []
    predictions = []
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions.extend(y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        smape_val = smape(y_test, y_pred)
        mae_scores.append(mae)
        smape_scores.append(smape_val)
    results[name] = {
        "MAE": np.mean(mae_scores),
        "SMAPE": np.mean(smape_scores)
    }
    all_predictions[name] = predictions

# Sonuçları Görselleştirme
results_df = pd.DataFrame(results).T
results_df.plot(kind="bar", figsize=(10, 6), rot=0)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.show()

# Sonuç Tablosu
print(results_df)

# Özelliklerin Model Tahminleri ile Karşılaştırılması
models_to_visualize = {"Random Forest": rf_model, "Gradient Boosting": gbr_model}

for model_name, model in models_to_visualize.items():
    plt.figure(figsize=(20, 15))
    for i, column in enumerate(df_features.columns):
        plt.subplot(3, 3, i + 1)
        model.fit(df_features[[column]], labels)
        predictions = model.predict(df_features[[column]])
        plt.scatter(df_features[column], labels, alpha=0.6, label="True Values")
        plt.scatter(df_features[column], predictions, alpha=0.6, label="Predictions", color='red')
        plt.title(f"{model_name}: {column}")
        plt.xlabel(column)
        plt.ylabel("lbl2")
        plt.legend()
    plt.tight_layout()
    plt.show()

# Hata Matrislerinin Görselleştirilmesi
for model_name, model in models.items():
    all_true = []
    all_pred = []
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_true.extend(y_test)
        all_pred.extend(y_pred)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Yuvarlama ve Hata Matrisi Hesaplama
    rounded_true = np.round(all_true)
    rounded_pred = np.round(all_pred)
    cm = confusion_matrix(rounded_true, rounded_pred, labels=np.unique(rounded_true))

    # Hata Matrisi Görselleştirme
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(rounded_true))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()
