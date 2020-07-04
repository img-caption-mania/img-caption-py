import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
from tqdm import tqdm
import pickle

PATH = os.getcwd()
checkpoint_path = PATH + "/checkpoints/train"
BATCH_SIZE = 3
BUFFER_SIZE = 1000
EPOCHS = 20
embedding_dim = 256
units = 512
top_k = 200
vocab_size = top_k + 1
num_steps = 384 // BATCH_SIZE  # 384 / 64 = 6 # from 80% of data train

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

with open(PATH + "/all_caption.json", 'r') as f:
  annotations = json.load(f)


def collect_capImg(annotations, num_examples=480):

    # Store captions and image names in list
    all_captions = []
    all_img_name_vector = []

    for annot in annotations:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_image_path = PATH + '/dataset/{IMG}'.format(IMG=image_id)
        all_img_name_vector.append(full_image_path)
        all_captions.append(caption)

    # Shuffle captions and image_names together
    # Set a random state
    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=1)

    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    return train_captions, img_name_vector

def img_featExtract():
  # initiate image feature extractor model

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')

    new_input = image_model.input

    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    return image_features_extract_model

def load_image(image_path):
  
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def cache_feature(img_name_vector, image_features_extract_model):
  # - `store the resulting vector` `in a dictionary` (image_name --> feature_vector).
  # - you pickle the dictionary and save it to disk.

    encode_train = sorted(set(img_name_vector))
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
      load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16) # map functional programming

    for img, path in tqdm(image_dataset):
      batch_features = image_features_extract_model(img) # image features extractor model
      # print("batch_features, batch_features.shape[0], -1, batch_features.shape[3] in order",batch_features, batch_features.shape[0], -1, batch_features.shape[3])

      batch_features = tf.reshape(batch_features,
                                  (batch_features.shape[0], -1, batch_features.shape[3]))

      for bf, p in zip(batch_features, path): # bf : batch feature , p : path
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())


# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize_cap(train_captions):
    # Choose the top 5000 words from the vocabulary
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    return tokenizer

def vectorize_cap(tokenizer, train_captions):

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    return cap_vector


# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap

def load_npFile(img_name_train, cap_train, BATCH_SIZE, BUFFER_SIZE):

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
              map_func, [item1, item2], [tf.float32, tf.int32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim) 64, 64, 256

    # hidden shape == (batch_size, hidden_size)  64, 512
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size) (64, 1, 512)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)) # vector_size 512 + vector_size 512

    # attention_weights shape == (batch_size, 64, 1) # from 512 --> 1 fc
    # (64, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)
    # print("attention_weights: ",attention_weights)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features # multiplication feature x attention
    # context_vector:  Tensor("rnn__decoder_1/bahdanau_attention_1/mul:0", shape=(64, 64, 256), dtype=float32)
    context_vector = tf.reduce_sum(context_vector, axis=1) # vector sum
    # context_vector:  Tensor("rnn__decoder_11/bahdanau_attention/Sum:0", shape=(64, 256), dtype=float32)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model): # init CNN_Encoder(embedding_dim)
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # (8, 8, 2048)
        # (64, 2048)
        # shape after fc == (batch_size, 64, embedding_dim)
        # (64, 256)
        self.fc = tf.keras.layers.Dense(embedding_dim) # embedding_dim = 256

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model): # init RNN_Decoder(embedding_dim, units, vocab_size)
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units                          # hidden units 512

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) # (5001, 256)

    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # embedding --> concat \w context_vector attention --> GRU --> fc1 \w output (to predict words) 512 --> reshape x (to predict words) --> fc1 \w x (to predict words) 5001 (vocab classification) 
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)  # 256 + hidden_size(context_vector)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)  # output to predicted words and state to next GRU layer unit
    # output:  Tensor("rnn__decoder_2/gru_2/transpose_1:0", shape=(64, 1, 512), dtype=float32)
    # state:  Tensor("rnn__decoder_2/gru_2/while:4", shape=(64, 512), dtype=float32)    

    # shape == (batch_size, max_length, hidden_size) # maximum_length of caption sentence = 13, hidden_sz = 512
    x = self.fc1(output)
    # x fc1 output:  Tensor("rnn__decoder_2/dense_13/BiasAdd:0", shape=(64, 1, 512), dtype=float32)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))
    # x reshape:  Tensor("rnn__decoder_2/Reshape:0", shape=(64, 512), dtype=float32)

    # output shape == (batch_size * max_length, vocab) # vocab = 201 need to be adjust to own vocab size
    x = self.fc2(x)
    # x fc2:  Tensor("rnn__decoder_2/dense_14/BiasAdd:0", shape=(64, 201), dtype=float32)

    return x, state, attention_weights    # x for predicted words class , state for GRU next hidden layer, attention_weights ???

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units)) # 64 x 512


def loss_function(real, pred, loss_object):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(img_tensor, target, decoder, encoder, tokenizer, loss_object, optimizer): # input from img tensor, n target word captions
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0]) # zeros(64x512)

  # word index <start> -> 2 
  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1) # decide input
  
  # print("target.shape[0]: ", target.shape[0])
  # target.shape[0]:  64 (batch size) 64 x 13
  # print("dec_input: ", dec_input)
  # dec_input:  Tensor("ExpandDims:0", shape=(64, 1), dtype=int32)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)  # 64 x 64 x 2048 --> 64 x 64 x 256
      # print("target.shape[1]: ", target.shape[1])
      # target.shape[1]:  13

      for i in range(1, target.shape[1]): # loop over the target 1-13 times (1 word 1 loop)
          # passing the features through the decoder
          # overwrite hidden from attention
          predictions, hidden, _ = decoder(dec_input, features, hidden)
          # print(i)

          loss += loss_function(target[:, i], predictions, loss_object)
          # print("target[:, i]: ", target[:, i])

          # using teacher forcing # overwrite new decide_input
          dec_input = tf.expand_dims(target[:, i], 1)

  # backpropagation
  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss

def evaluate(image, max_length, decoder, encoder, tokenizer, image_features_extract_model):
    # np zeros 13x64
    attention_plot = np.zeros((max_length, attention_features_shape))

    # tf zeros 1x512
    hidden = decoder.reset_state(batch_size=1)

    # load img resize n normalize 299x299x3 (`-1` - `1`) entry
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

def main():

  # collect cap img into variable  
  train_captions, img_name_vector = collect_capImg(annotations)
  # init img feature extractor model
  image_features_extract_model = img_featExtract()
  # cek summary feature extractor model
  # image_features_extract_model.summary()

  # done caching you can comment this one below
  cache_feature(img_name_vector, image_features_extract_model)
  
  tokenizer = tokenize_cap(train_captions)
  cap_vector = vectorize_cap(tokenizer, train_captions)
  max_length = calc_max_length(tokenizer.texts_to_sequences(train_captions))
  print(max_length)

  # Create training and validation sets using an 80-20 split '384 - 96'
  img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                      cap_vector,
                                                                      test_size=0.2,
                                                                      random_state=0)

  print('img_name_train: ', len(img_name_train), 'cap_train: ', len(cap_train), 'img_name_val: ', len(img_name_val), 'cap_val: ', len(cap_val))

  dataset = load_npFile(img_name_train, cap_train, BATCH_SIZE, BUFFER_SIZE)

  encoder = CNN_Encoder(embedding_dim)
  decoder = RNN_Decoder(embedding_dim, units, vocab_size)

  optimizer = tf.keras.optimizers.Adam()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')

  ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)

  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

  start_epoch = 0

  if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

  # adding this in a separate cell because if you run the training cell
  # many times, the loss_plot array will be reset
  loss_plot = []

  for epoch in range(start_epoch, EPOCHS):
      start = time.time()
      total_loss = 0

      for (batch, (img_tensor, target)) in enumerate(dataset): # 384 / 64 = 6 step or 6 batch
          batch_loss, t_loss = train_step(img_tensor, target, decoder, encoder, tokenizer, loss_object, optimizer)
          total_loss += t_loss

          if batch % 100 == 0:
              print ('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
      # storing the epoch end loss value to plot later
      loss_plot.append(total_loss / num_steps)

      if epoch % 5 == 0:
        ckpt_manager.save()

      print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                          total_loss/num_steps))
      print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

  plt.plot(loss_plot)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Plot')
  plt.show()

  # captions on the validation set
  rid = np.random.randint(0, len(img_name_val))
  image = img_name_val[rid]
  real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
  result, attention_plot = evaluate(image, max_length, decoder, encoder, tokenizer, image_features_extract_model)

  print ('Real Caption:', real_caption)
  print ('Prediction Caption:', ' '.join(result))
  plot_attention(image, result, attention_plot)

  image_path = PATH + '/data_test/IC5.jpg'
  # image_extension = image_url[-4:]
  # image_path = tf.keras.utils.get_file('image'+image_extension,
  #                                      origin=image_url)

  result, attention_plot = evaluate(image_path, max_length, decoder, encoder, tokenizer, image_features_extract_model)
  print ('Prediction Caption:', ' '.join(result))
  plot_attention(image_path, result, attention_plot)
  # opening the image
  Image.open(image_path)

  image_path = PATH + '/data_test/IC4.jpg'
  # image_extension = image_url[-4:]
  # image_path = tf.keras.utils.get_file('image'+image_extension,
  #                                      origin=image_url)

  result, attention_plot = evaluate(image_path, max_length, decoder, encoder, tokenizer, image_features_extract_model)
  print ('Prediction Caption:', ' '.join(result))
  plot_attention(image_path, result, attention_plot)
  # opening the image
  Image.open(image_path)

  image_path = PATH + '/data_test/IC3.JPG'
  # image_extension = image_url[-4:]
  # image_path = tf.keras.utils.get_file('image'+image_extension,
  #                                      origin=image_url)

  result, attention_plot = evaluate(image_path, max_length, decoder, encoder, tokenizer, image_features_extract_model)
  print ('Prediction Caption:', ' '.join(result))
  plot_attention(image_path, result, attention_plot)
  # opening the image
  Image.open(image_path)

  image_path = PATH + '/data_test/IC2.jpg'
  # image_extension = image_url[-4:]
  # image_path = tf.keras.utils.get_file('image'+image_extension,
  #                                      origin=image_url)

  result, attention_plot = evaluate(image_path, max_length, decoder, encoder, tokenizer, image_features_extract_model)
  print ('Prediction Caption:', ' '.join(result))
  plot_attention(image_path, result, attention_plot)
  # opening the image
  Image.open(image_path)

  image_path = PATH + '/data_test/IC1.jpg'
  # image_extension = image_url[-4:]
  # image_path = tf.keras.utils.get_file('image'+image_extension,
  #                                      origin=image_url)

  result, attention_plot = evaluate(image_path, max_length, decoder, encoder, tokenizer, image_features_extract_model)
  print ('Prediction Caption:', ' '.join(result))
  plot_attention(image_path, result, attention_plot)
  # opening the image
  Image.open(image_path)

if __name__ == "__main__":
    main()
