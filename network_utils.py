import keras


def ConvBlock(input_tensor, filters, kernel_size, padding, padding_type='valid'):
  if padding != 0:
    padded = keras.layers.ZeroPadding2D(padding)(input_tensor)
  else:
    padded = input_tensor

  conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                             padding=padding_type)(padded)

  bn = keras.layers.BatchNormalization()(conv)
  activated = keras.layers.LeakyReLU(0.01)(bn)

  return activated


def InceptionBlock(input_tensor, n_fmaps_1x1, n_fmaps_3x3):
  conv1x1 = ConvBlock(input_tensor, n_fmaps_1x1, 1, 0)
  conv3x3 = ConvBlock(input_tensor, n_fmaps_3x3, 3, 1)
  return keras.layers.concatenate([conv1x1, conv3x3])