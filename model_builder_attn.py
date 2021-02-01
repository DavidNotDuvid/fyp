import keras.layers
import keras.models
import tensorflow as tf

CONST_DO_RATE = 0.5

option_dict_conv = {"activation": "relu", "border_mode": "same"}
option_dict_bn = {"mode": 0, "momentum" : 0.9}


# returns a core model from gray input to 64 channels of the same size
def get_core(dim1, dim2):
    #query_input = tf.keras.Input(shape=(None,), dtype='int32')
    #value_input = tf.keras.Input(shape=(None,), dtype='int32')
    query_input = tf.keras.Input(shape=(dim1, dim2, 1), dtype='int32')
    value_input = tf.keras.Input(shape=(dim1, dim2, 1), dtype='int32')
    # Embedding lookup.
    token_embedding = tf.keras.layers.Embedding(1000,64)
# Query embeddings of shape [batch_size, Tq, dimension].
    query_embeddings = token_embedding(query_input)
# Value embeddings of shape [batch_size, Tv, dimension].
    value_embeddings = token_embedding(value_input)

    # CNN layer.
    cnn_layer = tf.keras.layers.Conv2D(filters=100,
                                       kernel_size=4,padding='same')
    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)
    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding])
# Reduce over the sequence axis to produce encodings of shape [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling2D()(query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling2D()(query_value_attention_seq)
# Concatenate query and document encodings to produce a DNN input layer.

    x = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])
    
    #x = keras.layers.Input(shape=(dim1, dim2, 1))

    a = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(x)  
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)

    a = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(a)
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)

    
    y = keras.layers.MaxPooling2D()(a)

    b = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(y)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)

    b = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(b)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)

    
    y = keras.layers.MaxPooling2D()(b)

    c = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(y)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)

    c = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(c)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)

    
    y = keras.layers.MaxPooling2D()(c)

    d = keras.layers.Convolution2D(512, 3, 3, **option_dict_conv)(y)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)

    d = keras.layers.Convolution2D(512, 3, 3, **option_dict_conv)(d)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)

    
    # UP

    d = keras.layers.UpSampling2D()(d)

    y = keras.layers.merge.concatenate([d, c], axis=3)

    e = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(y)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(e)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = keras.layers.UpSampling2D()(e)

    
    y = keras.layers.merge.concatenate([e, b], axis=3)

    f = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(y)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(f)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = keras.layers.UpSampling2D()(f)

    
    y = keras.layers.merge.concatenate([f, a], axis=3)

    y = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)

    y = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)

    return [x, y]


def get_model_3_class(dim1, dim2, activation="softmax"):
    
    [x, y] = get_core(dim1, dim2)

    y = keras.layers.Convolution2D(3, 1, 1, **option_dict_conv)(y)

    if activation is not None:
        y = keras.layers.Activation(activation)(y)

    model = keras.models.Model(x, y)
    
    return model
