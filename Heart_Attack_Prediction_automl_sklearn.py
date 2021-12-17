from evalml.automl import AutoMLSearch
import evalml
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount("/content/drive/")
df = pd.read_csv("/content/drive/kongkea/heart.csv")
df = df.drop(["oldpeak", "slp", "thall"], axis=1)
df.head()
df.shape
df.isnull().sum()
df.corr()
sns.heatmap(df.corr())
plt.figure(figsize=(20, 10))
plt.title("Patients' age")
plt.xlabel("Age")
sns.countplot(x="age", data=df)
plt.figure(figsize=(20, 10))
plt.title("Patients' sex")
sns.countplot(x="sex", data=df)
cp_data = df["cp"].value_counts().reset_index()
cp_data["index"][3] = "asymptomatic"
cp_data["index"][2] = "non-anginal"
cp_data["index"][1] = "atyppical"
cp_data["index"][0] = "typical"
cp_data
plt.figure(figsize=(20, 10))
plt.title("Patients' chest pain")
sns.barplot(x=cp_data["index"], y=cp_data["cp"])
ecg_data = df["restecg"].value_counts().reset_index()
ecg_data["index"][0] = "normal"
ecg_data["index"][1] = "abnormal"
ecg_data["index"][
    2
] = "probable/definite"
ecg_data
plt.figure(figsize=(20, 10))
plt.title("ECG")
sns.barplot(x=ecg_data["index"], y=ecg_data["restecg"])
sns.pairplot(df, hue="output", data=df)
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(df["trtbps"], kde=True, color="magenta")
plt.xlabel("Stable Blood Pressure (mmHg)")
plt.subplot(1, 2, 2)
sns.distplot(df["thalachh"], kde=True, color="teal")
plt.xlabel("Max Heart Rate (bpm)")
plt.figure(figsize=(10, 10))
sns.distplot(df["chol"], kde=True, color="red")
plt.xlabel("Cholestrol")
df.head()

scale = StandardScaler()
scale.fit(df)
df = scale.transform(df)
df = pd.DataFrame(
    df,
    columns=[
        "age",
        "sex",
        "cp",
        "trtbps",
        "chol",
        "fbs",
        "restecg",
        "thalachh",
        "exng",
        "caa",
        "output",
    ],
)
df.head()
x = df.iloc[:, :-1]
x
y = df.iloc[:, -1:]
y

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=101
)

lbl = LabelEncoder()
encoded_y = lbl.fit_transform(y_train)
logreg = LogisticRegression()
logreg = LogisticRegression()
logreg.fit(x_train, encoded_y)
Y_pred1

encoded_ytest = lbl.fit_transform(y_test)
Y_pred1 = logreg.predict(x_test)
lr_conf_matrix = confusion_matrix(encoded_ytest, Y_pred1)
lr_acc_score = accuracy_score(encoded_ytest, Y_pred1)
lr_conf_matrix
print(lr_acc_score * 100, "%")

tree = DecisionTreeClassifier()
tree.fit(x_train, encoded_y)
ypred2 = tree.predict(x_test)
encoded_ytest = lbl.fit_transform(y_test)
tree_conf_matrix = confusion_matrix(encoded_ytest, ypred2)
tree_acc_score = accuracy_score(encoded_ytest, ypred2)
tree_conf_matrix
print(tree_acc_score * 100, "%")

rf = RandomForestClassifier()
rf.fit(x_train, encoded_y)
ypred3 = rf.predict(x_test)
rf_conf_matrix = confusion_matrix(encoded_ytest, ypred3)
rf_acc_score = accuracy_score(encoded_ytest, ypred3)
rf_conf_matrix
print(rf_acc_score * 100, "%")

error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, encoded_y)
    pred = knn.predict(x_test)
    error_rate.append(np.mean(pred != encoded_ytest))
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, 40),
    error_rate,
    color="blue",
    linestyle="dashed",
    marker="o",
    markerfacecolor="red",
    markersize=10,
)
plt.xlabel("K Vlaue")
plt.ylabel("Error rate")
plt.title("Check value of K")
plt.show()
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(x_train, encoded_y)
ypred4 = knn.predict(x_test)
knn_conf_matrix = confusion_matrix(encoded_ytest, ypred4)
knn_acc_score = accuracy_score(encoded_ytest, ypred4)
knn_conf_matrix
print(knn_acc_score * 100, "%")

svm = svm.SVC()
svm.fit(x_train, encoded_y)
ypred5 = svm.predict(x_test)
svm_conf_matrix = confusion_matrix(encoded_ytest, ypred5)
svm_acc_score = accuracy_score(encoded_ytest, ypred5)
svm_conf_matrix
print(svm_acc_score * 100, "%")
model_acc = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "K Nearest Neighbor",
            "SVM",
        ],
        "Accuracy": [
            lr_acc_score * 100,
            tree_acc_score * 100,
            rf_acc_score * 100,
            knn_acc_score * 100,
            svm_acc_score * 100,
        ],
    }
)
model_acc = model_acc.sort_values(by=["Accuracy"], ascending=False)
model_acc

adab = AdaBoostClassifier(
    base_estimator=svm,
    n_estimators=100,
    algorithm="SAMME",
    learning_rate=0.01,
    random_state=0,
)
adab.fit(x_train, encoded_y)
ypred6 = adab.predict(x_test)
adab_conf_matrix = confusion_matrix(encoded_ytest, ypred6)
adab_acc_score = accuracy_score(encoded_ytest, ypred6)
adab_conf_matrix
print(adab_acc_score * 100, "%")
adab.score(x_train, encoded_y)
adab.score(x_test, encoded_ytest)

model_acc
param_grid = {
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "penalty": ["none", "l1", "l2", "elasticnet"],
    "C": [100, 10, 1.0, 0.1, 0.01],
}
grid1 = GridSearchCV(LogisticRegression(), param_grid)
grid1.fit(x_train, encoded_y)
grid1.best_params_
logreg1 = LogisticRegression(C=0.01, penalty="l2", solver="liblinear")
logreg1.fit(x_train, encoded_y)
logreg_pred = logreg1.predict(x_test)
logreg_pred_conf_matrix = confusion_matrix(encoded_ytest, logreg_pred)
logreg_pred_acc_score = accuracy_score(encoded_ytest, logreg_pred)
logreg_pred_conf_matrix
print(logreg_pred_acc_score * 100, "%")
n_neighbors = range(1, 21, 2)
weights = ["uniform", "distance"]
metric = ["euclidean", "manhattan", "minkowski"]
grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(
    estimator=knn, param_grid=grid, n_jobs=-1, cv=cv, scoring="accuracy", error_score=0
)
grid_search.fit(x_train, encoded_y)
grid_search.best_params_
knn = KNeighborsClassifier(
    n_neighbors=12, metric="manhattan", weights="distance")
knn.fit(x_train, encoded_y)
knn_pred = knn.predict(x_test)
knn_pred_conf_matrix = confusion_matrix(encoded_ytest, knn_pred)
knn_pred_acc_score = accuracy_score(encoded_ytest, knn_pred)
knn_pred_conf_matrix
print(knn_pred_acc_score * 100, "%")
kernel = ["poly", "rbf", "sigmoid"]
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ["scale"]
grid = dict(kernel=kernel, C=C, gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(
    estimator=svm, param_grid=grid, n_jobs=-1, cv=cv, scoring="accuracy", error_score=0
)
grid_search.fit(x_train, encoded_y)
grid_search.best_params_

svc = SVC(C=0.1, gamma="scale", kernel="sigmoid")
svc.fit(x_train, encoded_y)
svm_pred = svc.predict(x_test)
svm_pred_conf_matrix = confusion_matrix(encoded_ytest, svm_pred)
svm_pred_acc_score = accuracy_score(encoded_ytest, svm_pred)
svm_pred_conf_matrix
print(svm_pred_acc_score * 100, "%")
logreg = LogisticRegression()
logreg = LogisticRegression()
logreg.fit(x_train, encoded_y)
Y_pred1
lr_conf_matrix
print(lr_acc_score * 100, "%")
options = ["Disease", "No Disease"]
fig, ax = plt.subplots()
im = ax.imshow(lr_conf_matrix, cmap="Set3", interpolation="nearest")
ax.set_xticks(np.arange(len(options)))
ax.set_yticks(np.arange(len(options)))
ax.set_xticklabels(options)
ax.set_yticklabels(options)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(options)):
    for j in range(len(options)):
        text = ax.text(
            j, i, lr_conf_matrix[(i, j)], ha="center", va="center", color="black"
        )
ax.set_title("Confusion Matrix of Logistic Regression Model")
fig.tight_layout()
plt.xlabel("Model Prediction")
plt.ylabel("Ground truth")
plt.show()
print("Model accuracy ", lr_acc_score * 100, "%")

pickle.dump(logreg, open("heart.pkl", "wb"))
df = pd.read_csv("/content/drive/kongkea/heart.csv")
df.head()
x = df.iloc[:, :-1]
x
y = df.iloc[:, -1:]
y = lbl.fit_transform(y)
y

X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(
    x, y, problem_type="binary"
)
evalml.problem_types.ProblemTypes.all_problem_types

automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type="binary")
automl.search()
automl.rankings
automl.best_pipeline
best_pipeline = automl.best_pipeline
automl.describe_pipeline(automl.rankings.iloc[0]["id"])
best_pipeline.score(X_test, y_test, objectives=[
                    "auc", "f1", "Precision", "Recall"])
automl_auc = AutoMLSearch(
    X_train=X_train,
    y_train=y_train,
    problem_type="binary",
    objective="auc",
    additional_objectives=["f1", "precision"],
    max_batches=1,
    optimize_thresholds=True,
)
automl_auc.search()
automl_auc.rankings
automl_auc.describe_pipeline(automl_auc.rankings.iloc[0]["id"])
best_pipeline_auc = automl_auc.best_pipeline
best_pipeline_auc.score(X_test, y_test, objectives=["auc"])
best_pipeline.save("heart_model.pkl")
model = automl.load("heart_model.pkl")
model.predict_proba(X_test)
