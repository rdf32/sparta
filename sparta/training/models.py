import numpy as np
import tensorflow as tf


# models
def unet_model(config):
    inputs = tf.keras.layers.Input(tuple(config['shape'][0]))
    
    # encoder
    enc_block1 = encoder_block(inputs, config['filters'], config['batch_norm'], dropout_prob=config['dropouts'][0])
    enc_block2 = encoder_block(enc_block1[0], config['filters']*2, config['batch_norm'], dropout_prob=config['dropouts'][1])
    enc_block3 = encoder_block(enc_block2[0], config['filters']*4, config['batch_norm'], dropout_prob=config['dropouts'][2])
    enc_block4 = encoder_block(enc_block3[0], config['filters']*8, dropout_prob=config['dropouts'][3])
    
    # bridge
    bridge = conv_block(enc_block4[0], config['filters']*16, config['batch_norm'], dropout_prob=config['dropouts'][4])
    
    # decoder      
    dec_block4 = decoder_block(bridge, enc_block4[1], config['filters']*8, config['batch_norm'], dropout_prob=config['dropouts'][5])
    dec_block3 = decoder_block(dec_block4, enc_block3[1], config['filters']*4, config['batch_norm'], dropout_prob=config['dropouts'][6])
    dec_block2 = decoder_block(dec_block3, enc_block2[1], config['filters']*2, config['batch_norm'], dropout_prob=config['dropouts'][7])
    dec_block1 = decoder_block(dec_block2, enc_block1[1], config['filters'], config['batch_norm'], dropout_prob=config['dropouts'][8])

    # mutli-class classification
    if config['n_classes'] == 2:
        conv10 = tf.keras.layers.Conv2D(1, 1, padding='same')(dec_block1)
        output = tf.keras.layers.Activation('sigmoid', dtype='float32')(conv10)
    else:
        conv10 = tf.keras.layers.Conv2D(config['n_classes'], 1, padding='same')(dec_block1)
        output = tf.keras.layers.Activation('softmax', dtype='float32')(conv10)
    
    # create model object
    model = tf.keras.Model(inputs=inputs, outputs=output, name='Unet-detector')

    return model

def conv_block(inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
    conv1 = tf.keras.layers.SeparableConv2D(n_filters, 3, padding='same', depthwise_initializer="he_normal", pointwise_initializer="he_normal")(inputs)
    if batch_norm:
        conv1 = tf.keras.layers.BatchNormalization(axis=3)(conv1)
    conv1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)
    conv2 = tf.keras.layers.SeparableConv2D(n_filters, 3, padding='same', depthwise_initializer="he_normal", pointwise_initializer="he_normal")(conv1)
    if batch_norm:
        conv2 = tf.keras.layers.BatchNormalization(axis=3)(conv2)
    conv2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2)
    if dropout_prob > 0:
        conv2 = tf.keras.layers.Dropout(dropout_prob)(conv2)
    return conv2
   
def encoder_block(inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
    skip_connection = conv_block(inputs, n_filters, batch_norm, dropout_prob)
    next_layer = tf.keras.layers.SeparableConv2D(n_filters, 3, strides=2, padding='same')(skip_connection)
    return next_layer, skip_connection

def decoder_block(expansive_input, contractive_input, n_filters, batch_norm=False, dropout_prob=0):
    up = tf.keras.layers.Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)
    return conv_block(tf.keras.layers.concatenate([up, contractive_input], axis=3), n_filters, batch_norm, dropout_prob)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model 
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :self.d_model]
        return x

def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

def ffn(x, d_model, dropout=0.1):
    lin_x = tf.keras.layers.LayerNormalization()(x)
    lin_x = tf.keras.layers.Dense(d_model*4, activation='relu')(lin_x)
    lin_x = tf.keras.layers.Dense(d_model, activation='relu')(lin_x)
    lin_x = tf.keras.layers.Dropout(dropout)(lin_x)
    return tf.keras.layers.Add()([x, lin_x])

def self_attention(x, num_heads, d_model, dropout):
    norm_x = tf.keras.layers.LayerNormalization()(x)
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)(
        query=norm_x, 
        key=norm_x,
        value=norm_x,
        use_causal_mask=False)
    return tf.keras.layers.Add()([x, attn_output])

def encoder_layer(x, num_heads, d_model, dropout):
    x = self_attention(x, num_heads, d_model, dropout)
    x = ffn(x, d_model)
    return x

def transformer_encoder(config):
    inputs = tf.keras.Input(shape=(config['eseq_len'], config['embed_dim']))
    x = PositionalEmbedding(d_model=config['embed_dim'])(inputs)
    x = tf.keras.layers.Dropout(config['dropout'])(x)
    for i in range(config['num_elayers']):
        x = encoder_layer(x, config['num_heads'], config['embed_dim'], config['dropout'])
    outputs =  tf.keras.layers.Dense(config['n_classes'], activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='TransformerEncoder')

def transformer_encoder_mlp(config):
    inputs = tf.keras.Input(shape=(config['eseq_len'], config['embed_dim']))
    x = PositionalEmbedding(d_model=config['embed_dim'])(inputs)
    x = tf.keras.layers.Dropout(config['dropout'])(x)
    for i in range(config['num_elayers']):
        x = encoder_layer(x, config['num_heads'], config['embed_dim'], config['dropout'])
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs =  tf.keras.layers.Dense(config['n_classes'], activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='TransformerEncoderMLP')
