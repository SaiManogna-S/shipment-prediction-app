# install libraries

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, roc_auc_score
import joblib

# load dataset
data = pd.read_csv("train.csv")
print("Dataset Info:\n")
print(data.info())
print("\nMissing values:\n", data.isnull().sum())

# remove duplicates
print("Duplicates before:", data.duplicated().sum())
data.drop_duplicates(inplace=True)
print("Duplicates after:", data.duplicated().sum())

# handle outliers (IQR)
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = np.where(data[col] < lower, lower,
                         np.where(data[col] > upper, upper, data[col]))
print("Outliers capped")

# standardize categorical text
cat_cols = data.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    data[col] = data[col].str.strip().str.lower()

# visualize distributions
num_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols.remove("Reached.on.Time_Y.N")
sns.set(style="whitegrid", palette="Set2")
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(data[col], kde=True)
    plt.title(f"{col} Distribution")
    plt.show()

# countplots for categorical
for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=data, x=col, order=data[col].value_counts().index)
    plt.title(f"{col} Countplot")
    plt.xticks(rotation=45)
    plt.show()

# boxplots and KDE by target
target = "Reached.on.Time_Y.N"
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=target, y=col, data=data)
    plt.title(f"{col} vs {target}")
    plt.show()
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=data, x=col, hue=target, fill=True)
    plt.title(f"{col} Distribution by {target}")
    plt.show()

# correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data[num_cols+[target]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# class balance
print("Class distribution:\n", data[target].value_counts())
sns.countplot(x=target, data=data)
plt.title("Target Class Distribution")
plt.show()

# encoding
encode_data = data.copy()
le = LabelEncoder()
encode_data['Product_importance'] = le.fit_transform(encode_data['Product_importance'])
encode_data['Gender'] = le.fit_transform(encode_data['Gender'])
encode_data = pd.get_dummies(encode_data, columns=['Warehouse_block','Mode_of_Shipment'], drop_first=True)
encode_data = encode_data.astype(int)

# scaling
scaler = StandardScaler()
num_cols = ['Customer_care_calls','Customer_rating','Cost_of_the_Product','Prior_purchases','Discount_offered','Weight_in_gms']
encode_data[num_cols] = scaler.fit_transform(encode_data[num_cols])

# feature engineering
encode_data['Cost_to_Weight_ratio'] = encode_data['Cost_of_the_Product'] / encode_data['Weight_in_gms']
encode_data.replace([np.inf,-np.inf],np.nan,inplace=True)
encode_data.fillna(encode_data.median(), inplace=True)

# split
X = encode_data.drop('Reached.on.Time_Y.N', axis=1)
y = encode_data['Reached.on.Time_Y.N']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# smote
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# models
models = {
    "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "SVM": SVC(C=0.5, kernel='rbf', probability=True, random_state=42),
    "XGBoost": XGBClassifier(learning_rate=0.05, n_estimators=500, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.5, use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(learning_rate=0.05, n_estimators=500, max_depth=6, num_leaves=20, subsample=0.8, colsample_bytree=0.8, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=5, verbose=False, random_state=42)
}

# evaluation
comparison = []
for name, model in models.items():
    print(f"\nTraining {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)
    comparison.append({"Model":name,"Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1,"ROC-AUC":roc_auc})
    print(classification_report(y_val, y_pred))

comparison_df = pd.DataFrame(comparison).sort_values(by="ROC-AUC",ascending=False)
print("\nModel Comparison:\n", comparison_df)

# save best model
best_model = models["XGBoost"]
joblib.dump(best_model,"best_xgboost_model.pkl")
joblib.dump(X_train.columns.tolist(),"feature_columns.pkl")
print("Model saved successfully!")
