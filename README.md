# Fake-news-detection
#logistic_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print("\n--- Logistic Regression ---")
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(X_train_vectorized, y_train)

y_test_pred_lr = lr_classifier.predict(X_test_vectorized)
print(classification_report(y_test, y_test_pred_lr))
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_lr):.4f}")
#naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

print("\n--- Naive Bayes ---")
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

y_test_pred_nb = nb_classifier.predict(X_test_vectorized)
print(classification_report(y_test, y_test_pred_nb))
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_nb):.4f}")
#knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

print("\n--- K-Nearest Neighbors ---")
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_vectorized, y_train)

y_test_pred_knn = knn_classifier.predict(X_test_vectorized)
print(classification_report(y_test, y_test_pred_knn))
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_knn):.4f}")
#svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

print("\n--- Support Vector Machine ---")
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_vectorized, y_train)

y_test_pred_svm = svm_classifier.predict(X_test_vectorized)
print(classification_report(y_test, y_test_pred_svm))
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_svm):.4f}")
#decision_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

print("\n--- Decision Tree ---")
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_vectorized, y_train)

y_test_pred_dt = dt_classifier.predict(X_test_vectorized)
print(classification_report(y_test, y_test_pred_dt))
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_dt):.4f}")
#random_forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("\n--- Random Forest ---")
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_vectorized, y_train)

y_test_pred_rf = rf_classifier.predict(X_test_vectorized)
print(classification_report(y_test, y_test_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_rf):.4f}")
#neural_network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("\n--- Neural Network ---")
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=100, input_length=maxlen))
model.add(Dropout(0.2))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_padded, y_train_keras, epochs=10, batch_size=32, validation_split=0.2)
#Dependencies File (requirements.txt)
scikit-learn
tensorflow
numpy
pandas
matplotlib
seaborn







