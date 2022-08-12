import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa

# To suppress a warning when saving LeakyReLU
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, Layer, LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential, activations
from tensorflow import matmul
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score
from tensorflow.keras.constraints import UnitNorm


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

print("Splitting data...")
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=False, random_state=42)

# Create "clean" set for training (remove malicious subflows)
clean_indices_train = y_train[y_train['Anomaly'] == 0].index
X_train_clean = X_train.loc[clean_indices_train]

clean_indices_test = y_test[y_test['Anomaly'] == 0].index
X_test_clean = X_test.loc[clean_indices_test]

# In[10]:
# Tied Weights require custom layer
#https://medium.com/@lmayrandprovencher/building-an-autoencoder-with-tied-weights-in-keras-c4a559c529a2
class DenseTranspose(Layer):
    def __init__(self, dense, activation=None, **kwargs):
            self.dense = dense
            self.activation = activations.get(activation)
            super().__init__(**kwargs)
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias",
                                      shape=[self.dense.input_shape[-1]],
                                      initializer="zeros")
        super().build(batch_input_shape)
    def call(self, inputs):
        z = matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)
    def get_config(self):
       config = super(DenseTranspose, self).get_config()
       config.update({"dense": self.dense})
       return config

# In[2]:

# Auto encoder parameters
print("Initializing autoencoder...")
nb_epoch = 700
batch_size = 32
input_dim = X_train.shape[1]
hidden_dim = input_dim - 1
latent_dim = np.ceil(input_dim / 2)

# Options
model_name = 'autoencoder_5G_model15.tf'

#act1 = "relu"
#act2 = "linear"
act1 = act2 = LeakyReLU()
#encoder_constraint = decoder_constraint = None

encoder_constraint = UnitNorm(axis=0)
decoder_constraint = UnitNorm(axis=1)

opt = tf.keras.optimizers.Adam()
opt = tfa.optimizers.Lookahead(opt)

# Base Encoder 
hidden_1 = Dense(hidden_dim, 
                 activation=act1, 
                 input_shape=(input_dim,), 
                 kernel_constraint=encoder_constraint)
latent = Dense(latent_dim, 
               activation=act1, 
               kernel_constraint=encoder_constraint)

# Base Decoder 
hidden_2 = Dense(hidden_dim, activation=act1, kernel_constraint=decoder_constraint) 
out = Dense(input_dim, activation=act2, kernel_constraint=decoder_constraint)

# Dense Transpose Decoder (Tied Weights)
#hidden_2 = DenseTranspose(latent, activation=act1)
#out = DenseTranspose(hidden_1, activation=act2)

# Model
autoencoder = Sequential()
autoencoder.add(hidden_1)
autoencoder.add(latent)
autoencoder.add(hidden_2)
autoencoder.add(out)
autoencoder.summary()

# In[3]:
# Compile and Run model
print("Compiling autoencoder...")

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer=opt)


# Save checkpoint to upload the best model for testing
cp = ModelCheckpoint(filepath=model_name,
                     save_best_only=True,verbose=0)


#tb = TensorBoard(log_dir='./logs',
#                 histogram_freq=0,
#                 write_graph=True,
#                 write_images=True)

# Parameter helps prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)


history = autoencoder.fit(X_train_clean, X_train_clean,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test_clean, X_test_clean),
                    callbacks=[cp, early_stop]).history

#w = autoencoder.layers[0].get_weights()[0]
#w_t = w.T
#print(np.dot(w_t,w).round(2))
# In[4]:
# Plot loss against epochs
plt.plot(history['loss'], 'b', label='Training loss')
plt.plot(history['val_loss'], 'r', label='Validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mae]')
plt.show()

# In[5]:
# Load best model and find threshold
if act1 == "relu":
    autoencoder = load_model(model_name)
else:
    autoencoder = load_model(model_name, custom_objects={"act1": LeakyReLU(), "act2": LeakyReLU()})

X_train_pred = autoencoder.predict(X_train_clean)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train_clean), axis=1)
threshold = np.max(train_mae_loss)
print("Reconstuction error threshold: ", threshold)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("Number of samples")
plt.show()

# In[6]

# Calculate threshold by accounting for standard deviation
mean = np.mean(train_mae_loss, axis=0)
sd = np.std(train_mae_loss, axis=0)
num_sd = 3 # 3 standard deviations is about 99.7% of data 

final_list = [x for x in train_mae_loss if (x > mean - num_sd * sd)] 
final_list = [x for x in final_list if (x < mean + num_sd * sd)]
print("max value after removing 3*std:", np.max(final_list))
sd_threshold = np.max(final_list)
print("number of packets removed:", (len(train_mae_loss) - len(final_list)))
print("number of packets before removal:", len(train_mae_loss))

# In[7]:
# X_test can be replaced with live data

# Make predictions for X_test and calculate the difference 
test_x_predictions = autoencoder.predict(X_test) 
test_mae_loss = np.mean(np.abs(test_x_predictions - X_test), axis=1)

# Returns number of malicious (1) and normal (0) data points in test set

accuracy_score(y_test, [1 if s > sd_threshold else 0 for s in test_mae_loss])

# In[8]:
# Graph depicts threshold line and location of normal and malicious data

data = [test_mae_loss, y_test]
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

# In[9]:
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