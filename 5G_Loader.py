import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score
from tensorflow_addons.optimizers import Lookahead
from tensorflow.keras.optimizers import Adam

# In[1]:
# See 5G_Extractor.py
print("Loading 5G Features...")

path = 'C:\\Users\\Michael\\Dropbox\\Backup\\Michael\\Shared\\Documents\\VTEC\\5G Code\\features\\'
file = 'combined_5G_pcaps_features.csv'
df = pd.read_csv(path+file)
df.info()

features = df.iloc[0:,:-1].columns
target = ['Anomaly']

# In[2]:

# Anomaly Generator
def generate_mal_subflows(num_mal, pkts_sec, pkt_size_min, pkt_size_max, gradient):
    mal_subflows = []
    for i in range(num_mal):
        mal_subflows.append(mal_subflow(pkts_sec, pkt_size_min, pkt_size_max, gradient))
    mal_subflows = pd.DataFrame(mal_subflows)
    mal_subflows.columns = df.columns
    # Concatenate normal subflows and malicious subflows
    data = [df, mal_subflows]
    return pd.concat(data).sample(frac=1) # Shuffle and return

def mal_subflow(pkts_sec, pkt_size_min, pkt_size_max, gradient):
    mal_features = []
    dur = 6
    num_pkts = pkts_sec * dur

    # Simple ICMP flood with same size packets (bytes)
    pkt_size = pkt_size_max
    if gradient:
        pkt_size = np.random.randint(pkt_size_min, pkt_size_max)
    total_size = pkt_size * num_pkts
    total_size /= 1e6 # MB
    bits_sec = (total_size * 8)/dur # MB to Mbit/s
    
    mal_features.append(pkts_sec)
    mal_features.append(bits_sec)
    mal_features.append(pkt_size) 
    mal_features.append(0) # no standard deviation for uniform sizes
    for i in range(5):
        mal_features.append(pkt_size)
    mal_features.append(1) # Mark as anomaly
    
    # TODO: Need to generate anomalous TCP flags and TTL values
    return pd.Series(mal_features)

# Synthetic Flood Generation

# Average nominal: 22.64 MBit/sec (@1427 packets/sec)
# Anomalies: 24 MBit/sec - 120 MBit/sec (@3000 packets/sec)
pkts_sec = 3000
pkt_size_min = 1000
pkt_size_max = 5000 # 65,535 bytes max (IPv4)

# Number of malicious subflows to generate
num_mal = np.ceil(df.shape[0] / 5).astype(int) # 20% of clean subflows

# Gradient generates each flow using packet size between min and max
# Otherwise all packets are max size
# Note: Packet sizes are always equal in each synthetic flow
gradient = True 

dirty_subflows = generate_mal_subflows(num_mal, pkts_sec, pkt_size_min, pkt_size_max, gradient)
X = dirty_subflows[features]
y = dirty_subflows[target]

# In[3]:

# Load best model and find threshold
# See models.txt
model_name = "models\\autoencoder_5G_model_8_ddos.tf"

autoencoder = load_model(model_name)

# If errors when loading, use custom object with load_model() 
# custom_objects={"act1": LeakyReLU(), "act2": LeakyReLU(), "opt":Lookahead(Adam())}

X_pred_clean = autoencoder.predict(df[features])
clean_mae_loss = np.mean(np.abs(X_pred_clean - df[features]), axis=1)
threshold = np.max(clean_mae_loss)
print("Reconstuction error threshold: ", threshold)

# Calculate threshold by accounting for standard deviation
mean = np.mean(clean_mae_loss, axis=0)
sd = np.std(clean_mae_loss, axis=0)
num_sd = 3

# '2*sd' = ~97.5%, '1.76 = ~96%', '1.64 = ~95%'
final_list = [x for x in clean_mae_loss if (x > mean - num_sd * sd)] 
final_list = [x for x in final_list if (x < mean + num_sd * sd)]
sd_threshold = np.max(final_list)
print("max value after removing 3*std:", sd_threshold)
print("number of packets removed:", (len(clean_mae_loss) - len(final_list)))
print("number of packets before removal:", len(clean_mae_loss))

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

# In[5]:
# Confusion Matrix heat map

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