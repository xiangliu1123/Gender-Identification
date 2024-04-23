import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv('all_csv\\preProcess_df.csv')

# Split features and target variable
X = data.drop(columns=['Gender'])
y = data['Gender']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# ======================================================================================================================
# I. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_predict_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_predict_knn))
print("KNN Classification Report:")
print(classification_report(y_test, y_predict_knn))

# ======================================================================================================================
# II. Artificial Neural Network (ANN)
ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
ann.fit(X_train, y_train)
y_predict_ann = ann.predict(X_test)
print("ANN Accuracy:", accuracy_score(y_test, y_predict_ann))
print("ANN Classification Report:")
print(classification_report(y_test, y_predict_ann))

# ======================================================================================================================
# III. Naïve Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_predict_nb = nb.predict(X_test)
print("Naïve Bayes Accuracy:", accuracy_score(y_test, y_predict_nb))
print("Naïve Bayes Classification Report:")
print(classification_report(y_test, y_predict_nb))

# ======================================================================================================================
# IV. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_predict_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_predict_lr))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_predict_lr))

# ======================================================================================================================
# V. Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_predict_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_predict_dt))
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_predict_dt))

# ======================================================================================================================
# VI. Support Vector Machine (SVM)
svm = SVC(kernel='linear', random_state=42)  # Using a linear kernel initially
svm.fit(X_train, y_train)
y_predict_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_predict_svm))
print("SVM Classification Report:")
print(classification_report(y_test, y_predict_svm))

# ======================================================================================================================
# VII. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_predict_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_predict_rf))
print("Random Forest Classification Report:")
print(classification_report(y_test, y_predict_rf))

# ======================================================================================================================
