import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score

# In[1]:
# See 5G_Extractor.py
print("Loading 5G Features...")

path = '/smallwork/m.hackett_local/data/ashley_pcaps/captures/features/'
file = 'combined_5G_pcaps_features.csv'
df = pd.read_csv(path+file)
df.info()

features = df.iloc[0:,:-1].columns
target = ['Malicious']

X = df[features]
y = df[target]

# Create "clean" set (remove malicious subflows)
clean_indices = y[y['Malicious'] == 0].index
X_clean = X.loc[clean_indices]

# In[2]:
# Load best model and find threshold

model_name = "autoencoder_5G_model9.tf"

autoencoder = load_model(model_name)
#autoencoder = load_model(model_name, custom_objects={"act1": LeakyReLU(), "act2": LeakyReLU()})

X_pred_clean = autoencoder.predict(X_clean)
clean_mae_loss = np.mean(np.abs(X_pred_clean - X_clean), axis=1)
threshold = np.max(clean_mae_loss)
print("Reconstuction error threshold: ", threshold)

# In[3]:
# Graph depicts threshold line and location of normal and malicious data

X_pred = autoencoder.predict(X) 
mae_loss = np.mean(np.abs(X_pred - X), axis=1)
 
data = [mae_loss, y]
error_df_test = pd.concat(data, axis=1)
error_df_test.columns=['Reconstruction_error','True_class']

error_df_test = error_df_test.reset_index()

groups = error_df_test.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, 
            marker='o', ms=3.5, linestyle='', 
            label= "Malicious" if name == 1 else "Normal") 
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

# In[4]:
#Confusion Matrix heat map

pred_y = [1 if e > threshold else 0 for e in error_df_test['Reconstruction_error'].values]
conf_matrix = confusion_matrix(error_df_test['True_class'], pred_y) 
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix,
            xticklabels=["Normal","Malicious"], 
            yticklabels=["Normal","Malicious"], 
            annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

#   TN | FP
#   -------
#   FN | TP

print(" accuracy:  ", accuracy_score(error_df_test['True_class'], pred_y))
print(" recall:    ", recall_score(error_df_test['True_class'], pred_y))
print(" precision: ", precision_score(error_df_test['True_class'], pred_y))
print(" f1-score:  ", f1_score(error_df_test['True_class'], pred_y))