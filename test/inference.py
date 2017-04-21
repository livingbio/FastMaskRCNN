#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # noqa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import tensorflow.contrib.slim as slim
import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1
from libs.nets.train_utils import _get_variables_to_train, get_var_list_to_restore
from libs.layers import roi_decoder


class MaskRCNN(object):
    """
    Do inference for the pretrained MaskRCNN. (Backbone ResNet-50)
    """

    def __init__(self, gpu_mem=0.5):
        self.resnet50 = resnet_v1.resnet_v1_50
        self.FLAGS = tf.app.flags.FLAGS
        self.gpu_mem = gpu_mem

    def build_model(self, print_model=False):
        """Build up MaskRCNN Architecture and initialize variables."""
        print('\nBuilding model...')

        ## data
        image_in = tf.placeholder(tf.float32, [None, None, 3])
        ih = tf.placeholder(tf.int32)
        iw = tf.placeholder(tf.int32)

        image, _, _ = coco_preprocess.preprocess_image(image_in, gt_boxes=None,
                                                       gt_masks=None, is_training=False)

        ##  network
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, end_points = self.resnet50(image, 1000, is_training=False)
            end_points['inputs'] = image

        if print_model:
            for x in sorted(end_points.keys()):
                print (x, end_points[x].name, end_points[x].shape)

        pyramid = pyramid_network.build_pyramid('resnet50', end_points)

        self.outputs = pyramid_network.build_heads(pyramid, ih, iw, num_classes=81,
                                                   base_anchors=15, is_training=False)
        self.variables_to_train = _get_variables_to_train()
        init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

        # Each image will sample and select 52 ROIs - default value (N=52)
        # final boxes: (N, 4)
        # classes: (N,)
        # scores: (N,)
        box = self.outputs['P5']['refined']['box']
        cls_prob = tf.nn.softmax(self.outputs['P5']['refined']['cls'])
        rois = self.outputs['P5']['roi']['box']
        final_boxes, classes, scores = roi_decoder(box, cls_prob, rois, ih, iw)

        self.graph_inputs = [image_in, ih, iw]
        self.graph_outputs = [final_boxes, classes, scores]
        print('\nDone building model.')

        print('\nCreating session and initializing variables...')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_mem, 
                                    allow_growth=True,
                                    )
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                     allow_soft_placement=True))
        global_step = slim.create_global_step()
        self.sess.run(init_op)

    def restore_model(self, pretrained_model_path=None, print_model=False):
        """
        Restore pretrained model.
        pretrained_model_path: path to pretrained model or use FLAGS.pretrained_model
        """
        if pretrained_model_path:
            model_path = pretrained_model_path
        else:
            model_path = self.FLAGS.pretrained_model

        if tf.gfile.IsDirectory(model_path):
            checkpoint_path = tf.train.latest_checkpoint(model_path)
            vars_to_restore = self.variables_to_train
        else:
            checkpoint_path = model_path
            self.FLAGS.checkpoint_exclude_scopes='pyramid'
            self.FLAGS.checkpoint_include_scopes='resnet_v1_50'
            vars_to_restore = get_var_list_to_restore()

        if print_model:
            for var in vars_to_restore:
                print ('restoring ', var.name)

        try:
            restorer = tf.train.Saver(vars_to_restore)
            restorer.restore(self.sess, checkpoint_path)
            print ('\nRestored %d(%d) vars from %s' %(
                len(vars_to_restore), len(tf.global_variables()),
                checkpoint_path ))
            print('\nDone restoring')
        except Exception as e:
            print(e)
            print ('\nChecking your params %s' %(checkpoint_path))
            print('\nFail restoring. Using randomly initialized parameters.')

    def transform_classes(self, cid_list):
        """Mapping from category ids to real object classes"""

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

    def inference(self, data_list, trans_cls=True, draw_box=True, n_box=10):

        image_data = []
        if isinstance(data_list[0], str) or isinstance(data_list[0], unicode):
            # list of filenames
            for fn in data_list:
                img = np.array(Image.open(fn))
                img_h, img_w, _ = img.shape
                image_data.append((img, img_h, img_w))

        elif type(data_list[0]).__module__ == np.__name__:
            # list of numpy arrays
            for img in data_list:
                img_h, img_w, _ = img.shape
                image_data.append((img, img_h, img_w))

        else:
            print('\nThe input "data_list" must be a list of either strings or numpy arrays.')
            return

        image_in, ih, iw = self.graph_inputs
        inference_results = []
        for ind, data in enumerate(image_data):
            result = {}
            image_data, img_h, img_w = data
            feed_dict = {image_in: image_data, ih: img_h, iw: img_w}

            # inference
            boxes, clss, scs = self.sess.run(self.graph_outputs,
                                             feed_dict=feed_dict)

            # transform category id to real object name
            if trans_cls:
                clss = self.transform_classes(clss)

            result['boxes'] = boxes
            result['classes'] = clss
            result['scores'] = scs

            inference_results.append(result)

            if draw_box:
                img_box = self.draw_boxes(image_data, boxes, clss, scs, n_box)
                img_box.save('image_box_{}.png'.format(str(ind)))

        return inference_results

    def draw_boxes(self, img, boxes, classes, scores, n_box=10,
                   line_col=(0, 255, 0), text_col=(255, 255, 255), width=0):
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)

        rank = sorted([(sc, ind) for ind, sc in enumerate(scores)], reverse=True)
        for sc, ind in rank[:n_box]:
            x1, y1, x2, y2 = boxes[ind]
            draw.line((x1, y1, x1, y2), fill=line_col, width=width)
            draw.line((x1, y2, x2, y2), fill=line_col, width=width)
            draw.line((x2, y2, x2, y1), fill=line_col, width=width)
            draw.line((x2, y1, x1, y1), fill=line_col, width=width)

            draw.text((x1, y1), classes[ind] + ':%.2f' % sc, fill=text_col)

        del draw

        return im


if __name__ == '__main__':
    clf = MaskRCNN()
    clf.build_model()
    clf.restore_model()
    results = clf.inference(['./test/dog.jpeg', 'room.jpeg'])
    import pdb;pdb.set_trace()
