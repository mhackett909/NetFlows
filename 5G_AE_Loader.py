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
target = ['Anomaly']

X = df[features]
y = df[target]

# Remove original malicious subflows
clean_indices = y[y['Anomaly'] == 0].index
X_clean = X.loc[clean_indices]

# In[2]:
# Load best model and find threshold

model_name = "autoencoder_5G_model1.tf"

autoencoder = load_model(model_name)
#autoencoder = load_model(model_name, custom_objects={"act1": LeakyReLU(), "act2": LeakyReLU()})

X_pred_clean = autoencoder.predict(X_clean)
clean_mae_loss = np.mean(np.abs(X_pred_clean - X_clean), axis=1)
threshold = np.max(clean_mae_loss)
print("Reconstuction error threshold: ", threshold)

# Calculate threshold by accounting for standard deviation
mean = np.mean(clean_mae_loss, axis=0)
sd = np.std(clean_mae_loss, axis=0)
num_sd = 2

# '2*sd' = ~97.5%, '1.76 = ~96%', '1.64 = ~95%'
final_list = [x for x in clean_mae_loss if (x > mean - num_sd * sd)] 
final_list = [x for x in final_list if (x < mean + num_sd * sd)]
sd_threshold = np.max(final_list)
print("max value after removing 2*std:", sd_threshold)
print("number of packets removed:", (len(clean_mae_loss) - len(final_list)))
print("number of packets before removal:", len(clean_mae_loss))

# In[3]:
def generate_mal_subflows(num_mal, num_pkts, pkt_size):
    mal_subflows = []
    for i in range(num_mal):
        mal_subflows.append(mal_subflow(num_pkts, pkt_size))
    mal_subflows = pd.DataFrame(mal_subflows)
    X_clean['Anomaly'] = 0
    mal_subflows.columns = X_clean.columns
    # Concatenate normal subflows and malicious subflows
    data = [X_clean, mal_subflows]
    return pd.concat(data).sample(frac=1)

def mal_subflow(num_pkts, pkt_size):
    mal_features = []
    dur = 5
    pkts_sec = num_pkts/dur
    mal_features.append(pkts_sec)
    # Simple ICMP flood with same size packets (bytes)
    total_size = pkt_size * num_pkts
    total_size /= 1e6
    bits_sec = (total_size * 8)/dur
    mal_features.append(bits_sec)
    mal_features.append(pkt_size) 
    mal_features.append(0) # no standard deviation for uniform sizes
    for i in range(5):
        mal_features.append(pkt_size)
    mal_features.append(0) # No TCP flags for ICMP ping
    mal_features.append(64) # Arbitrary TTL
    mal_features.append(1) # Mark as anomaly
    return pd.Series(mal_features)

# THIS IS WHERE CUSTOM MAL SUBFLOWS ARE CREATED
pkt_size = 5000 # 64 - 65,535 bytes (IPv4)
# Number of malicious subflows to generate
num_mal = np.ceil(X_clean.shape[0] / 5).astype(int) # 20% of clean subflows
# Packets per second should be high, at least 6000 
num_pkts = np.random.randint(30000,35000) # Uses 5 second duration for calculation

dirty_subflows = generate_mal_subflows(num_mal, num_pkts, pkt_size)
X = dirty_subflows[features]
y = dirty_subflows[target]

# In[4]:

# Graph depicts threshold line and location of normal and malicious data
X_pred = autoencoder.predict(X) 
test_mae_loss = np.mean(np.abs(X_pred - X), axis=1)
 
data = [test_mae_loss, y]
error_df_test = pd.concat(data, axis=1)
error_df_test.columns=['Reconstruction_error','True_class']

error_df_test = error_df_test.reset_index()

groups = error_df_test.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, 
            marker='o', ms=3.5, linestyle='', 
            label= "Anomaly" if name == 1 else "Normal") 
ax.hlines(sd_threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

# In[4]:
#Confusion Matrix heat map

pred_y = [1 if e > sd_threshold else 0 for e in error_df_test['Reconstruction_error'].values]
conf_matrix = confusion_matrix(error_df_test['True_class'], pred_y) 
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix,
            xticklabels=["Normal","Anomaly"], 
            yticklabels=["Normal","Anomaly"], 
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