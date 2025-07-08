import keras 
import tensorflow as tf
from keras.layers import Conv2D , Conv2DTranspose , Input , Flatten , Dense , Lambda, Reshape
import keras.layers
from keras.models import Model 
from keras.datasets import mnist
from keras import backend as k 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import animation
import numpy as np
from ipywidgets import interact, FloatSlider

(x_train , y_train) , (x_test , y_test) = mnist.load_data()

### Data Normalization 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255

### Reshaping Images 
img_width = x_train.shape[1]
img_height = x_train.shape[2]

print(f"MNIST Images Shape : {img_width , img_height }")

num_channels = 1 ### MNIST dataset ==> grey scale images ==> 1 channel 
x_train = x_train.reshape(x_train.shape[0] , img_height , img_width, num_channels) 
x_test = x_test.reshape(x_test.shape[0] , img_height , img_width , num_channels) 

input_shape = (img_height , img_width , num_channels)

### Visualizing the Data 

plt.figure(1)
plt.subplot(221)
plt.imshow(x_train[15][:,:,0])

plt.subplot(222)
plt.imshow(x_train[560][:,:,0])

plt.subplot(223)
plt.imshow(x_train[3500][:,:,0])

plt.subplot(224)
plt.imshow(x_train[35000][:,:,0])
plt.show()

#@@@@@@@@@@@@@@@@@@@@@
#@@@@@@ Encoder @@@@@@
#@@@@@@@@@@@@@@@@@@@@@

latent_dim= 3


input_img = Input(shape=input_shape , name="enc_input")
x = Conv2D(32 , 3 , padding="same" , activation="relu")(input_img)
x = Conv2D(64 , 3 , padding="same" , activation="relu" , strides=(2,2))(x)
x = Conv2D(64 , 3 , padding="same" , activation="relu")(x)
#x = Conv2D(64 , 3 , padding="same" , activation="relu")(x)
x = Conv2D(64 , 3 , padding="same" , activation="relu")(x)

conv_shape = k.int_shape(x) ### Shape to be provided to the decoder
print(f"Shape to be provided to the decoder : {conv_shape}")
x = Flatten()(x)
x = Dense(32,activation='relu')(x)

### 
z_mu = Dense(latent_dim , name="latent_mu")(x) ### Mean value of encoded input
z_sigma = Dense(latent_dim , name="latent_sigma")(x) ### Standard deviation of encoded input

### Reparameterization Trick : 

def sample_z(args):
    z_mu , z_sigma = args
    eps = k.random_normal(shape=(k.shape(z_mu)[0],k.int_shape(z_mu)[1]))
    return z_mu + k.exp(z_sigma/2)*eps

z = Lambda(sample_z , output_shape=(latent_dim, ) , name='z')([z_mu,z_sigma])

encoder = Model(input_img , [z_mu,z_sigma,z] , name="encoder")
print(encoder.summary())


#@@@@@@@@@@@@@@@@@@@@@
#@@@@@@ Decoder @@@@@@
#@@@@@@@@@@@@@@@@@@@@@

### the decoder takes the latent vector z as input
decoder_input = Input(shape=(latent_dim,) , name="dec_input")
x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3] , activation="relu")(decoder_input) ## decoder will receive the output of the flattening layer 
x = Reshape((conv_shape[1] , conv_shape[2] , conv_shape[3]))(x)
x = Conv2DTranspose(32,3,padding='same' , activation="relu" , strides=(2,2))(x)
x = Conv2DTranspose(num_channels , 3 , padding='same' , activation='sigmoid' , name="decoder_deconv")(x)


decoder = Model(decoder_input , x , name="decoder")
print(decoder.summary())

z_decoded = decoder(z) ### Reconstructed Image 
print("shape of the decoded Image : " , z_decoded.shape) ### Same shape as the input image 


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@ Building the Model @@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class VAE(keras.Model) : 
    def __init__(self,encoder,decoder,beta,**kwargs):
        super(VAE,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta 
        
    def compile(self , optimizer) : 
        super(VAE,self).compile()
        self.optimizer = optimizer
        self.total_loss_tracker =  tf.keras.metrics.Mean(name="Loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="Recon_Loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="KL_Loss")
        
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker ,
            self.kl_loss_tracker
            ]
    
    def train_step(self , data) : 
        if isinstance(data,tuple) : 
            data = data[0]
        
        with tf.GradientTape() as tape : 
            z_mean , z_sigma , z = self.encoder(data)
            recon = self.decoder(z)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, recon)))
            kl_loss = -0.5*tf.reduce_mean(tf.reduce_sum(1 + z_sigma - tf.square(z_mean) - tf.exp(z_sigma)))
            total_loss = recon_loss + self.beta*kl_loss
            
        grads = tape.gradient(total_loss , self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
    
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
    def call(self, inputs):
        """Required for model.__call__ and validation to work."""
        _, _, z = self.encoder(inputs)
        return self.decoder(z)
    
    
    
vae = VAE(encoder , decoder , beta=0.5)

vae.compile(tf.keras.optimizers.Adam())

vae.fit(x_train , epochs=10, batch_size=32 , validation_split=0.2)

vae.summary()

mu,_,_ = encoder.predict(x_test)


import plotly.express as px
import pandas as pd

df = pd.DataFrame(mu, columns=["dim1", "dim2", "dim3"])
df["label"] = y_test

fig = px.scatter_3d(df, x="dim1", y="dim2", z="dim3",
                    color=df["label"].astype(str), 
                    color_continuous_scale="brbg")

fig.update_traces(marker=dict(size=3))
fig.update_layout(title="Latent Space (3D)", margin=dict(l=0, r=0, b=0, t=30))
fig.show()

sample_vector = np.array([[2.7,1.7,-2]])
decoded_example = decoder.predict(sample_vector)
decoded_example_reshaepd = decoded_example.reshape(img_width , img_height)
plt.imshow(decoded_example_reshaepd)

###########################
# For a fixed value of dim3 
###########################
n = 20 
figure = np.zeros((img_width * n , img_height*n , num_channels))
grid_x = np.linspace(-3,3,n)
grid_y = np.linspace(-3,3,n)[::-1]
fixed_z3 = -1
for i , j in enumerate(grid_y) : 
    for k , l in enumerate(grid_x) : 
        z_sample = np.array([[l,j,fixed_z3]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(img_width , img_height , num_channels)
        figure[i*img_width : (i+1) * img_width,
               k*img_height : (k+1) * img_height] = digit
        
plt.figure(figsize=(8,8))
fig_shape = np.shape(figure)
figure = figure.reshape((fig_shape[0] , fig_shape[1]))
plt.imshow(figure , cmap="gnuplot2") 
plt.show()

##############################
# For any chosen value of dim3 
##############################

def plot_latent_slice(z3):
    n = 20
    figure = np.zeros((img_width * n, img_height * n, num_channels))
    grid_x = np.linspace(-5, 5, n)
    grid_y = np.linspace(-5, 5, n)[::-1]

    for i, y in enumerate(grid_y):
        for j, x in enumerate(grid_x):
            z_sample = np.array([[x, y, z3]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(img_width, img_height, num_channels)
            figure[i * img_width: (i + 1) * img_width,
                   j * img_height: (j + 1) * img_height] = digit

    plt.figure(figsize=(8, 8))
    if num_channels == 1:
        plt.imshow(figure.squeeze(), cmap   ='gray')
    else:
        plt.imshow(figure) #kasra f ydayk zawz ba33ed 3liya     aslan lfaza heki 9bal dharbet trend maa prianka w jones
    plt.title(f"Latent Dimension 3 = {z3:.2f}")
    plt.axis("off")
    plt.show()

interact(plot_latent_slice, z3=FloatSlider(min=-3.0, max=3.0, step=0.2, value=0.0));