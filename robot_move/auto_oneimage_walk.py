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

def remake_depthmap(dep, x = 0.3):
    change = dep[0,120:125,:,0].mean() - x
    dep = dep - change
    return dep

def percentaage_of_pixel_infer(dep, value=1):
    kek=dep[dep<=value]
    perc=(np.size(kek)*100)/np.size(dep)
    return(perc)

def refine_depth(dep,size=4):
    kernel = np.ones((size,size),np.uint8)
    gradient = cv.morphologyEx(dep, cv.MORPH_GRADIENT, kernel)
    return(gradient)

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

        # Use to load from npy file
        #net.load(model_data_path, sess)
        try:
            print('')
            print('- Model loaded')
            iterate = 0
            reset_servo(1000)
            reset_servo2()
            while True:
                print('')
                reset_delete(True)
                iterate = iterate + 1
                print('Start the ' +str(iterate) + 'th iteration')
                print(' - scan the front')
                take_one_photo()
                start_get_image_from_robot()
                # read imqge
                images = glob.glob(image_path + '/*.jpg')
                if len(images) != 0:
                    image = images[0]
                    print('  have got image, start depth prediction')
                    #img = Image.open(image)
                    image0 = cv2.imread(image)
                    img = cv2.resize(image0, (width,height))
                    img = np.array(img).astype('float32')
                    img = np.expand_dims(np.asarray(img), axis = 0)

                    pred = sess.run(net.get_output(), feed_dict={input_node: img})
                    #pred = remake_depthmap(pred)
                    os.remove(image)
                    deps = glob.glob(save_path+'/*.npy')
                    l = len(deps)
                    np.save(save_path+'/dep_'+ str(l+1)+'.npy', pred)
                    cv2.imwrite('keys/images/'+str(l+1) +'.jpg',image0)
                    print('    -- ' + str(l+1) + ' depth maps saved')

                print(' - make 3d point cloud')
                images = glob.glob('/keys/images/*.jpg')
                res = None
                imgfile = 'keys/images/'+str(1)+'.jpg'
                depfile = 'keys/deps/dep_'+str(1)+'.npy'
                res = add_image_demo(imgfile, depfile, res, NUM_PHOTO)
                #write_ply('keys/res/fin_'+str(iterate)+'.ply', res)

                print(' - model saved to fin_'+str(iterate)+'.ply')

                print(' robot go ahead and turn left')
                depth = np.load(depfile)
                image = cv2.imread('keys/images/'+str(1)+'.jpg')
                #dep = remake_depthmap(dep)
                """
                if depth.max()-depth.min() < 0.5 :
                    go_ahead_and_turn(0, 0.4)
                    print('distance is 0')
                else:
                    distance = depth[0,0:100,40:120,0].min()
                    go_ahead_and_turn(distance-0.2, 0)
                    print('distance is '+ str(distance))
                """

                distance , ress , seg= calcule_distance_from_seg(image, depth)
                #cv2.imwrite('keys/segs/'+str(iterate) +'rr.jpg',ress*50)
                #cv2.imwrite('keys/segs/'+str(iterate) +'ss.jpg',seg*20)
                cv2.imwrite('keys/segs/'+str(iterate) +'ii.jpg',image)
                np.save('keys/segs/'+str(iterate) +'dd.npy',depth.reshape(128,160))
                print(distance)
                if distance < 0.1:
                    go_ahead_and_turn(0, 1)
                elif distance > 0.1:
                    go_ahead_and_turn(min(distance, 0.4), 0)
                else:
                    go_ahead_and_turn(0, 0.1)
                print(' -- iteration finished')

        except KeyboardInterrupt:
            print('Robot stopped')
            return 0


def main():
    # Parse arguments
    ckptfile = 'F:/pfe/cnn_slam/FCRN-DepthPrediction-master/ckpt/NYU_FCRN.ckpt'
    imagefile = 'keys/new_image'
    depfile = 'keys/deps'
    # Predict the image
    print('Starting Tensorflow')
    predict(ckptfile, imagefile, depfile)

    os._exit(0)

if __name__ == '__main__':
    main()
