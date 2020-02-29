import tensorflow as tf

def deeplabv3(inputs, class_num, is_training=False):
    feature_map, low_level_feature = ResNet_50(inputs, True, is_training)

    encoder_feature = encoder(feature_map, 256, is_training, name='encoder')
    last_feature = decoder(low_level_feature, encoder_feature, class_num, 256, is_training, name='decoder')

    return last_feature

def encoder(cnn_fmap, depth, is_training=None, name=None):
    # def conv_2d(in_fmap, filter_size, in_ch, out_ch, stride, padding='SAME', name=None):
    # def atrous_conv2d(in_fmap, filter_size, in_ch, out_ch, rate, padding='SAME', name=None):
    with tf.variable_scope(name) as scope:
        _1x1conv = conv_2d(cnn_fmap, 1, 2048, depth, 1, name='_1x1conv')
        _3x3conv_6 = atrous_conv2d(in_fmap=cnn_fmap, filter_size=3, in_ch=2048, out_ch=depth, rate=6, name='_3x3conv_6')
        _3x3conv_12 = atrous_conv2d(cnn_fmap, 3, 2048, depth, 12, name='_3x3conv_12')
        _3x3conv_18 = atrous_conv2d(cnn_fmap, 3, 2048, depth, 18, name='_3x3conv_18')

        input_size = tf.shape(cnn_fmap)[1:3]
        # print(tf.shape(cnn_fmap)[1:3])
        # print(input_size)

        img_pooling = tf.reduce_mean(cnn_fmap, [1, 2])
        img_pooling = tf.reshape(img_pooling, [1, 1, 1, 2048])
        img_pooling = conv_2d(img_pooling, 1, 2048, depth, 1, name='img_pool_conv')
        img_pooling = tf.layers.batch_normalization(img_pooling, training=is_training)
        img_pooling = tf.nn.relu(img_pooling)
        img_pooling = tf.image.resize_bilinear(img_pooling, input_size, name='img_pooling')

        concat_feature = tf.concat([_1x1conv, _3x3conv_6, _3x3conv_12, _3x3conv_18, img_pooling], 3)
        concat_feature = conv_2d(concat_feature, 1, 1280, depth, 1, name='convcat_conv')
        concat_feature = tf.layers.batch_normalization(concat_feature, training=is_training)
        concat_feature = tf.nn.relu(concat_feature, name='concat_feature')

        return concat_feature

def decoder(low_feature, encoder_feature, classes, depth, is_training=None, name=None):
    with tf.variable_scope(name) as scope:
        input_size = tf.shape(low_feature)[1:3]

        low_feature = conv_2d(low_feature, 1, 256, depth, 1, name='low_feature_conv')
        low_feature = tf.layers.batch_normalization(low_feature, training=is_training)
        low_feature = tf.nn.relu(low_feature, name='low_feature')

        encoder_feature = tf.image.resize_bilinear(encoder_feature, input_size)

        _feature = tf.concat([low_feature, encoder_feature], 3)

        _feature = conv_2d(_feature, 3, 512, 256, 1, name='conv1')
        _feature = tf.layers.batch_normalization(_feature, training=is_training)
        _feature = tf.nn.relu(_feature)

        _feature = conv_2d(_feature, 3, 256, 256, 1, name='conv2')
        _feature = tf.layers.batch_normalization(_feature, training=is_training)
        _feature = tf.nn.relu(_feature)

        _feature = conv_2d(_feature, 3, 256, classes, 1, name='last_conv')
        _feature = tf.layers.batch_normalization(_feature, training=is_training)
        _feature = tf.nn.relu(_feature)

        return tf.image.resize_bilinear(_feature, (input_size[0] * 4, input_size [1] * 4))

# def conv_2d(in_fmap, filter_size, in_ch, out_ch, stride, padding='SAME', name=None):
# def bottle_block(in_fmap, in_ch, out_ch, stride=1, rate=None, is_training=False, downsampling=False, padding='SAME', name=None):
def ResNet_50(in_fmap, use_lowfeature=False, is_training=False):
    fmap = conv_2d(in_fmap, 7, 3, 64, stride=2, name='conv1')
    fmap = max_pooling(fmap, 3, 2, name='conv2_pool')

    fmap = bottle_block(fmap, 64, 64, stride=1, is_training=is_training, downsampling=True, is_first=True, name='conv2_1')
    fmap = bottle_block(fmap, 256, 64, stride=1, is_training=is_training, name='conv2_2')
    fmap = bottle_block(fmap, 256, 64, stride=1, is_training=is_training, name='conv2_3')

    if use_lowfeature != False:
        saved_lowfeature = fmap

    fmap = bottle_block(fmap, 256, 128, stride=2, is_training=is_training, downsampling=True, name='conv3_1')
    fmap = bottle_block(fmap, 512, 128, stride=1, is_training=is_training, name='conv3_2')
    fmap = bottle_block(fmap, 512, 128, stride=1, is_training=is_training, name='conv3_3')
    fmap = bottle_block(fmap, 512, 128, stride=1, is_training=is_training, name='conv3_4')

    fmap = bottle_block(fmap, 512, 256, stride=2, is_training=is_training, downsampling=True, name='conv4_1')
    fmap = bottle_block(fmap, 1024, 256, stride=1, is_training=is_training, name='conv4_2')
    fmap = bottle_block(fmap, 1024, 256, stride=1, is_training=is_training, name='conv4_3')
    fmap = bottle_block(fmap, 1024, 256, stride=1, is_training=is_training, name='conv4_4')
    fmap = bottle_block(fmap, 1024, 256, stride=1, is_training=is_training, name='conv4_5')
    fmap = bottle_block(fmap, 1024, 256, stride=1, is_training=is_training, name='conv4_6')

    fmap = bottle_block(fmap, 1024, 512, rate=2, is_training=is_training, downsampling=True, is_first=True, name='conv5_1')
    fmap = bottle_block(fmap, 2048, 512, rate=2, is_training=is_training, name='conv5_2')
    fmap = bottle_block(fmap, 2048, 512, rate=2, is_training=is_training, name='conv5_3')

    if use_lowfeature != None:
        return fmap, saved_lowfeature
    else:
        return fmap, _

# def conv_2d(in_fmap, filter_size, in_ch, out_ch, stride, padding='SAME', name=None):
# def atrous_conv2d(in_fmap, filter_size, in_ch, out_ch, rate, padding='SAME', name=None):

def bottle_block(in_fmap, in_ch, out_ch, stride=1, rate=None, is_training=False, downsampling=False, is_first=False, padding='SAME', name=None):
    if stride != None:
        conv_func = conv_2d
        param = 1
        first_param = stride
    elif rate != None:
        conv_func = atrous_conv2d
        param = rate
    else:
        raise

    fmap = conv_func(in_fmap, 1, in_ch, out_ch, first_param, padding, name+'_1conv')
    fmap = tf.layers.batch_normalization(fmap, training=is_training)
    fmap = tf.nn.relu(fmap, name=name+'_1ac')

    fmap = conv_func(fmap, 3, out_ch, out_ch, param, padding, name+'_2conv')
    fmap = tf.layers.batch_normalization(fmap, training=is_training)
    fmap = tf.nn.relu(fmap, name=name+'_2ac')

    fmap = conv_func(fmap, 1, out_ch, out_ch * 4, param, padding, name+'_3conv')
    fmap = tf.layers.batch_normalization(fmap, training=is_training)

    if downsampling == True:
        if is_first == True:
            d_stride = 1
        else:
            d_stride = 2

        in_fmap = conv_func(in_fmap, 1, in_ch, out_ch*4, d_stride, padding, name+'down')
        in_fmap = tf.layers.batch_normalization(in_fmap, training=is_training)


    fmap = tf.add(fmap, in_fmap)
    fmap = tf.nn.relu(fmap, name=name+'_3ac')

    return fmap

def atrous_conv2d(in_fmap, filter_size, in_ch, out_ch, rate, padding='SAME', name=None):
    filter_shape = [filter_size, filter_size, in_ch, out_ch]

    with tf.variable_scope(name) as scope:
        W = tf.get_variable(name='W', shape=filter_shape)
        b = tf.get_variable(name='b', shape=[out_ch])

        fmap = tf.nn.atrous_conv2d(value=in_fmap, filters=W, rate=rate, padding=padding, name=name)
        fmap = tf.nn.bias_add(fmap, b)

        return fmap

def conv_2d(in_fmap, filter_size, in_ch, out_ch, stride, padding='SAME', name=None):
    filter_shape = [filter_size, filter_size, in_ch, out_ch]
    strides = [1, stride, stride, 1]

    with tf.variable_scope(name) as scope:
        W = tf.get_variable(name='W', shape=filter_shape)
        b = tf.get_variable(name='b', shape=[out_ch])

        fmap = tf.nn.conv2d(in_fmap, filter=W, strides=strides, padding=padding, name=name)
        fmap = tf.nn.bias_add(fmap, b)

        return fmap

def max_pooling(in_fmap, filter_size, stride, padding='SAME', name=None):
    kernel_size = [1, filter_size, filter_size, 1]
    strides = [1, stride, stride, 1]

    with tf.variable_scope(name) as scope:
        return tf.nn.max_pool(value=in_fmap, ksize=kernel_size, strides=strides, padding=padding, name=name)
