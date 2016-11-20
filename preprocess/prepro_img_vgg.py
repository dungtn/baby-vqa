import os
import json
import h5py
import argparse
import numpy as np
from keras.preprocessing import image
from keras.applications import VGG19
from keras.applications.imagenet_utils import preprocess_input


def load_img(img_name):
  im = image.load_img(img_name, target_size=(224,224))
  im = image.img_to_array(im)
  im = np.expand_dims(im, axis=0)
  im = preprocess_input(im)
  return im


def get_batches(data, batch_size):
  return [data[i:i + batch_size] for i in xrange(0, len(data), batch_size)]


def main(params):
  # take the output features from the last pooling layer
  model = VGG19(include_top=False, weights='imagenet', input_tensor=None)

  json_file = json.load(open(params['input_json'], 'r'))
  img_list_train = json_file['unique_img_train']
  img_list_test = json_file['uniuqe_img_test']
  batch_size = params['batch_size']

  print "process %d training images..." % len(img_list_train)
  features_train = []
  for img_names in get_batches(img_list_train, batch_size):
    ims = []
    for img_name in img_names:
      im = load_img(os.path.join(params['image_root'], img_name))
      ims.append(im)
    features_train.extend(model.predict(np.vstack(ims), batch_size=batch_size))

  train_h5_file = h5py.File(params['out_name_train'], 'w')
  train_h5_file.create_dataset('images_train', data=features_train)

  print "process %d testing images..." % len(img_list_test)
  features_test = []
  for img_names in get_batches(img_list_test, batch_size):
    ims = []
    for img_name in img_names:
      im = load_img(os.path.join(params['image_root'], img_name))
      ims.append(im)
    features_test.extend(model.predict(np.vstack(ims), batch_size=batch_size))

  test_h5_file = h5py.File(params['out_name_test'], 'w')
  test_h5_file.create_dataset('images_test', data=features_test)


if __name__=='__main__':
  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', default='../data/vqa_data_prepro.json', help='path to the json file containing vocab and answers')
  parser.add_argument('--image_root', default='../data/', help='path to the image root')
  parser.add_argument('--batch_size', default=20, type=int, help='batch size')
  parser.add_argument('--out_name_train', default='../data/vqa_data_img_vgg_train.h5', help='output name for training set')
  parser.add_argument('--out_name_test', default='../data/vqa_data_img_vgg_test.h5', help='output name for testing set')

  args = parser.parse_args()
  params = vars(args)

  print 'parsed input parameters:'
  print json.dumps(params, indent = 2)
  main(params)
