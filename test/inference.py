#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # noqa
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1
from libs.nets.train_utils import _get_variables_to_train, get_var_list_to_restore
from libs.layers import roi_decoder


def restore_model():
    resnet50 = resnet_v1.resnet_v1_50
    FLAGS = tf.app.flags.FLAGS

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, 
                                    allow_growth=True,
                                    )
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                allow_soft_placement=True))
        global_step = slim.create_global_step()

        ## data
        image_in = tf.placeholder(tf.float32, [None, None, 3])
        ih = tf.placeholder(tf.int32)
        iw = tf.placeholder(tf.int32)

        image, gt_boxes, gt_masks = coco_preprocess.preprocess_image(image_in, gt_boxes=None, gt_masks=None, is_training=False)

        ##  network
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, end_points = resnet50(image, 1000, is_training=False)
            end_points['inputs'] = image

        for x in sorted(end_points.keys()):
            print (x, end_points[x].name, end_points[x].shape)

        pyramid = pyramid_network.build_pyramid('resnet50', end_points)
        # for p in pyramid:
        #   print (p, pyramid[p])

        outputs = pyramid_network.build_heads(pyramid, ih, iw, num_classes=81, base_anchors=15, is_training=False)
        variables_to_train = _get_variables_to_train()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess.run(init_op)

        ## restore pretrained model
        if FLAGS.pretrained_model:
            if tf.gfile.IsDirectory(FLAGS.pretrained_model):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model)
                vars_to_restore = variables_to_train
            else:
                checkpoint_path = FLAGS.pretrained_model
                FLAGS.checkpoint_exclude_scopes='pyramid'
                FLAGS.checkpoint_include_scopes='resnet_v1_50'
                vars_to_restore = get_var_list_to_restore()

            for var in vars_to_restore:
                print ('restoring ', var.name)

            try:
                restorer = tf.train.Saver(vars_to_restore)
                restorer.restore(sess, checkpoint_path)
                print ('Restored %d(%d) vars from %s' %(
                    len(vars_to_restore), len(tf.global_variables()),
                    checkpoint_path ))
            except:
                print ('Checking your params %s' %(checkpoint_path))
                raise

        print('Done restoring')

        box = outputs['P5']['refined']['box']
        cls_prob = tf.nn.softmax(outputs['P5']['refined']['cls'])
        rois = outputs['P5']['roi']['box']

        # Each image will sample and select 52 ROIs - default value (N=52)
        # final boxes: (N, 4)
        # classes: (N,)
        # scores: (N,)
        final_boxes, classes, scores = roi_decoder(box, cls_prob, rois, ih, iw)

    return final_boxes, classes, scores, sess, image_in, ih, iw


def transform_classes(cid_list):
    cid_to_cat = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
     6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
     11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
     15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow',
     21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack',
     26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee',
     31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat',
     36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket',
     40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife',
     45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich',
     50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza',
     55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant',
     60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop',
     65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave',
     70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock',
     76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush'}

    return [cid_to_cat[cid] for cid in cid_list]


def predict_boxes(filenames, trans_cls=True):
    final_boxes, classes, scores, sess, image_in, ih, iw = restore_model()

    results = {}
    for filename in filenames:
        results[filename] = {}

        # prepare data
        image_data = np.array(Image.open(filename))
        img_h, img_w, _ = image_data.shape
        feed_dict = {image_in: image_data, ih: img_h, iw: img_w}

        # inference
        boxes, clss, scs = sess.run([final_boxes, classes, scores],
                                    feed_dict=feed_dict)

        # transform category id to real object name
        if trans_cls:
            clss = transform_classes(clss)

        results[filename]['boxes'] = boxes
        results[filename]['classes'] = clss
        results[filename]['scores'] = scs

    return results


if __name__ == '__main__':
    results = predict_boxes(['TheMachine.png'])
    import pdb;pdb.set_trace()
