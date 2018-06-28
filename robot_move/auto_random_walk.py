import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import glob
from utils_realtime import *
from utils_walkdemo import *
import models


NUM_PHOTO = 1
if_segmentation = False
small_range =  True
ckptfile = 'F:/pfe/cnn_slam/FCRN-DepthPrediction-master/ckpt/NYU_FCRN.ckpt'
imagefile = 'keys/new_image'
depfile = 'keys/deps'

def predict(model_data_path, image_path, save_path):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)
        reset_delete(True,True)
        # Use to load from npy file
        #net.load(model_data_path, sess)
        print('')
        print('- Model loaded')
        iterate = 0
        reset_servo()
        reset_delete(True)
        try:
            while True:
                print('')
                iterate = iterate + 1
                print('Start the ' +str(iterate) + 'th iteration')
                print(' - scan the surrounding')
                if small_range:
                    scan_surrounding_small_range(NUM_PHOTO)
                else:
                    scan_surrounding(NUM_PHOTO)
                # read imqge
                images = glob.glob(image_path + '/*.jpg')
                dir_this_iter = save_path+'/dep'+str(iterate)
                if not os.path.exists(dir_this_iter):
                    os.makedirs(dir_this_iter)
                if len(images) != 0:
                    print('  have got images, start depth prediction')
                    for image in images:
                        #img = Image.open(image)
                        image0 = cv2.imread(image)
                        img = cv2.resize(image0, (width,height))
                        img = np.array(img).astype('float32')
                        img = np.expand_dims(np.asarray(img), axis = 0)

                        pred = sess.run(net.get_output(), feed_dict={input_node: img})
                        os.remove(image)
                        deps = glob.glob(dir_this_iter+'/*.npy')
                        l = len(deps)
                        np.save(dir_this_iter+'/dep_'+ str(l+1)+'.npy', pred)
                        cv2.imwrite('keys/images/'+ str(l+1) +'.jpg',image0)
                    print('    -- ' + str(l+1) + ' depth maps saved')

                print(' - make 3d point cloud')
                images = glob.glob('F:/pfe/segmentation/keys/images/*.jpg')
                res = None
                for i in range(NUM_PHOTO):
                    idx = i + 1
                    imgfile = 'keys/images/'+str(idx)+'.jpg'
                    depfile = dir_this_iter + '/dep_'+str(idx)+'.npy'
                    res, d0 = add_image_demo(imgfile, depfile,res, NUM_PHOTO,if_segmentation, small_range)
                if res is None:
                    write_ply('keys/res/fin_'+str(iterate)+'.ply', np.zeros((1,6)))
                else:
                    write_ply('keys/res/fin_'+str(iterate)+'.ply', res)
                np.save('keys/res/fin_'+str(iterate)+'.npy', res)

                print(' - model saved to fin_'+str(iterate)+'.ply')

                print(' robot go ahead and turn left')

                if res is None:
                    distance = 0
                else:
                    print('d0 is ', d0)
                    distance = distance_ahead(res) - d0 + 1.7
                print('distance is ',distance)
                go_ahead_and_turn(distance)

                if if_segmentation:
                    draw_boundbox_v2('keys/objs/index'+str(NUM_PHOTO-1)+'.npy')
                print(' -- iteration finished')

        except KeyboardInterrupt:
            print('Robot stopped')
            return 0


def main():
    # Parse arguments
    # Predict the image
    print('Starting Tensorflow')
    predict(ckptfile, imagefile, depfile)

    os._exit(0)

if __name__ == '__main__':
    main()
