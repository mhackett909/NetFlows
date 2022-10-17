import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
import os

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

path = 'C:\\Users\\Michael\\Dropbox\\Backup\\Michael\\Shared\\Documents\\VTEC\\5G Code\\features\\'
file = 'combined_5G_pcaps_features.csv'
df = pd.read_csv(path+file)
df.info()

features = df.iloc[0:,:-1].columns
target = ['Anomaly']

X = df[features]
y = df[target]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False, random_state=42)

# In[2]:

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

# In[3]:

# Auto encoder parameters
print("Initializing autoencoder...")
nb_epoch = 700
batch_size = 32
input_dim = X_train.shape[1]
hidden_dim = input_dim - 1
latent_dim = np.ceil(input_dim / 2)

# See models.txt
model_name = 'autoencoder_5G_model_8_ddos.tf'

#act1 = "relu"
#act2 = "linear"
act1 = act2 = LeakyReLU()
encoder_constraint = decoder_constraint = None

#encoder_constraint = UnitNorm(axis=0)
#decoder_constraint = UnitNorm(axis=1)

opt = tf.keras.optimizers.Adam()
#opt = tfa.optimizers.Lookahead(opt)

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

# In[4]:
# Compile and Run model
print("Compiling autoencoder...")

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer=opt)

# Save checkpoint to upload the best model for testing
model_path = 'models/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
cp = ModelCheckpoint(filepath=model_path+model_name,
                     save_best_only=True,verbose=0)

# Parameter helps prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    callbacks=[cp, early_stop]).history

# In[5]:
# Plot loss against epochs
plt.plot(history['loss'], 'b', label='Training loss')
plt.plot(history['val_loss'], 'r', label='Validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mae]')
plt.show()