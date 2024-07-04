import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

data = pd.read_csv('dataset.csv')

soil_type_label_encoder = LabelEncoder()
data['Soil Type'] = soil_type_label_encoder.fit_transform(data['Soil Type'])
crop_type_label_encoder = LabelEncoder()
data['Crop Type'] = crop_type_label_encoder.fit_transform(data['Crop Type'])
fert_type_label_encoder = LabelEncoder()
data['Fertilizer Name'] = fert_type_label_encoder.fit_transform(data['Fertilizer Name'])

X = data.drop(columns=['Fertilizer Name'])
y = data['Fertilizer Name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipeline_rf = make_pipeline(StandardScaler(), RandomForestClassifier())
pipeline_rf.fit(X_train, y_train)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = cross_val_score(pipeline_rf, X_train, y_train, cv=kf)

y_pred_rf = pipeline_rf.predict(X_test)
r2_score_rf = r2_score(y_test, y_pred_rf)
accuracy_score_rf = accuracy_score(y_test, y_pred_rf)

rf_model = pipeline_rf.named_steps['randomforestclassifier']
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance in Random Forest')
plt.show()

