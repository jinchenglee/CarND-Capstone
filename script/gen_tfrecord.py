import tensorflow as tf
import yaml
import os
import io
from object_detection.utils import dataset_util
from lxml import etree
import PIL.Image
import logging
import hashlib
from random import shuffle

MY_LABEL_DICT =  {
    "green" : 1,
    "red" : 2,
    "yellow" : 3,
    "off" : 4
    }

def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'): 
  img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    #print label_map_dict[obj['name']]
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):  

    annotationpath = os.path.join('../data', 'dataset_xml')

#    datalist = dataset_util.read_examples_list(os.path.join('../data', 'dataset_list'))
#    # Randomly shuffle the list
#    shuffle(datalist)
#    # Split into trainlist and testlist
#    split_marker = int(0.75*len(datalist))
#    trainlist = datalist[0:split_marker]
#    testlist = datalist[split_marker:-1]
#
#    list_file = open("trainlist", "w")
#    for item in trainlist:
#        list_file.write(str(item)+"\n")
#    list_file.close()
#
#    list_file = open("testlist", "w")
#    for item in testlist:
#        list_file.write(str(item)+"\n")
#    list_file.close()

    trainlist = dataset_util.read_examples_list(os.path.join('../data', 'trainlist'))
    testlist = dataset_util.read_examples_list(os.path.join('../data', 'testlist'))

    # Training record file gen
    writer = tf.python_io.TFRecordWriter("../data/train.tfrecord")
    for idx, filename in enumerate(trainlist):
        path = os.path.join(annotationpath, filename + '.xml')

        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        print("Gen Training: xml = ", data['filename'])
        tf_example = dict_to_tf_example(data, '../data', MY_LABEL_DICT,
                                      True, image_subdirectory='')
        writer.write(tf_example.SerializeToString())
    writer.close()

    # Test/Validation record file gen
    writer = tf.python_io.TFRecordWriter("../data/eval.tfrecord")
    for idx, filename in enumerate(testlist):
        path = os.path.join(annotationpath, filename + '.xml')

        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        print("Gen Testing: xml = ", data['filename'])
        tf_example = dict_to_tf_example(data, '../data', MY_LABEL_DICT,
                                      True, image_subdirectory='')
        writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    tf.app.run()
