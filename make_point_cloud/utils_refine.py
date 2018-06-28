import sys
import cv2
import numpy as np
import glob
import os
from utils import *
from math import sqrt
from numpy import *


newHeight = 128
newWidth = 160
seg_image_file = 'deeplab/tensorflow-deeplab-v3-master/images'
seg_image_list = 'deeplab/tensorflow-deeplab-v3-master/dataset/sample_images_list.txt'
seg_script = 'deeplab/tensorflow-deeplab-v3-master/inference.py'

seg_out = 'deeplab/tensorflow-deeplab-v3-master/dataset/inference_output'
seg_final = 'keys/seg'

dirs = []
dirs.append('keys')
dirs.append('keys/segs')
dirs.append('keys/deps')
dirs.append('keys/Ms')
dirs.append('keys/res')

def make_dir():
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return 0


def rigid_transform_3D(A, B):
    # Input: expects Nx3 matrix of points
    assert len(A) == len(B)
    N = A.shape[0]; # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.dot(transpose(AA),BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)
    t = -np.dot(R,centroid_A.T) + centroid_B.T
    M = np.zeros((3,4))
    M[0:3,0:3] = R
    M[:,3] = t
    return M

def projection_3d_2d_v0(res, T, K):
    T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)
    T = np.linalg.inv(T)
    T = T[0:3,:]
    # projection from the 3d points to 2d
    image = np.zeros((newHeight,newWidth,3),dtype=np.uint8)
    depth = np.zeros((newHeight,newWidth))
    res_out = np.zeros((len(res),6))
    t = 0
    for i in range(len(res)):
        u = np.array([res[i,0], res[i,1], res[i,2], 1])
        z = np.dot(T, u.T)
        z = np.dot(K,z.T)
        x = int(z[0]/z[2])
        y = int(z[1]/z[2])
        if 0<=x<newHeight and 0<=y<newWidth:
            image[x,y,:] = res[i,3:6]
            depth[x,y] = z[2]
        else:
            res_out[t,:] = res[i,:]
            t = t + 1
    #kernel = np.ones((5,5),np.uint8)
    #depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
    return image, depth, res_out[0:t, :]

def projection_3d_2d_v1(res, T, K):
    M = T
    T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)
    T = np.linalg.inv(T)
    T = T[0:3,:]
    # projection from the 3d points to 2d
    image = np.zeros((newHeight,newWidth,3),dtype=np.uint8)
    depth = np.zeros((newHeight,newWidth))
    res_out = np.zeros((len(res),6))
    t = 0
    for i in range(len(res)):
        u = np.array([res[i,0], res[i,1], res[i,2], 1])
        z = np.dot(T, u.T)
        z = np.dot(K,z.T)
        x = int(z[0]/z[2])
        y = int(z[1]/z[2])
        if 0<=x<newHeight and 0<=y<newWidth:
            if z[2] < 0 :
                res_out[t,:] = res[i,:]
                t = t + 1
            elif depth[x,y] == 0 and z[2] > 0:
                image[x,y,:] = res[i,3:6]
                depth[x,y] = z[2]
            else:
                if depth[x,y] > z[2] and abs(depth[x,y]-z[2]) > 0.3:
                    image[x,y,:] = res[i,3:6]
                    depth[x,y] = z[2]
                    temp = np.array([x,y,1]).T
                    out = np.dot(np.linalg.inv(K),(temp)*depth[x,y])
                    out = np.dot(M[0:3,0:3],out.reshape(3,1)) + M[:,3].reshape(1,3)
                    print(out)
                    res_out[t,0:3] = out[0,0:3]
                    print(res_out[t,0:3])
                    res_out[t,3:6] = image[x,y,:]
                    t = t + 1
                elif depth[x,y] < z[2] and abs(depth[x,y]-z[2]) > 0.3 :
                    res_out[t,:] = res[i,:]
                    t = t + 1
        else:
            res_out[t,:] = res[i,:]
            t = t + 1
    #kernel = np.ones((5,5),np.uint8)
    #depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
    return image, depth, res_out[0:t, :]


def judgement(im1, dep1, im2, dep2):
    threshold_img = 200
    threshold_dep = 0.5
    t = 0
    dst_img = 0
    dst_dep = 0
    dep1 = dep1.reshape((newHeight,newWidth))
    dep2 = dep2.reshape((newHeight,newWidth))
    for i in range(newHeight):
        for j in range(newWidth):
            if dep1[i,j] != 0 and dep2[i,j] != 0:
                dst_img += np.sum(np.abs(im1[i,j,:]-im2[i,j,:]))
                dst_dep += np.abs(dep1[i,j] - dep2[i,j])
                t = t + 1
    dst_img = dst_img/t
    dst_dep = dst_dep/t
    #print(dst_img,dst_dep)
    if dst_img>threshold_img or dst_dep>threshold_dep:
        return False
    else:
        return True

def combine_dep_known_rgb(img1, dep1, img2, dep2, kk = 300):
    # average dep1 and dep2 (for value non zero)
    # the rest will keep the value of dep1
    dep1 = dep1.reshape((newHeight,newWidth))
    dep2 = dep2.reshape((newHeight,newWidth))
    for i in range(newHeight):
        for j in range(newWidth):
            if dep1[i,j] != 0 and dep2[i,j] != 0:
                if np.sum(np.abs(img1[i,j,:]-img2[i,j,:])) < kk:
                    dep1[i,j] = (1.9*dep1[i,j]+0.1*dep2[i,j])/2
    return dep1.reshape((1,newHeight,newWidth,1))


def refine_v1(res, img, dep, T, K):
    img0, dep0 = projection_3d_2d(res, np.linalg.inv(T), K)
    if judgement(img0, dep0, img, dep):
        dep = combine_dep_known_rgb(img, dep, img0, dep0)

        image = np.zeros((newHeight,newWidth,3),dtype=np.uint8)
        depth = np.zeros((newHeight,newWidth))
        for i in range(len(res)):
            u = np.array([res[i,0], res[i,1], res[i,2], 1])
            z = np.dot(np.linalg.inv(T), u.T)
            z = np.dot(K,z)
            x = int(z[0]/z[2])
            y = int(z[1]/z[2])
            if 0<=x<newHeight and 0<=y<newWidth:
                if np.sum(np.abs(res[i,3:6]-img[x,y,:])) < 60:
                    d_t = (z[2]+dep[x,y])/2
                    img_p = np.array([x, y, 1])
                    aa = - np.dot(T[0:3,0:3].T, T[3,0:3]) + np.dot(T[0:3,0:3].T,np.dot(np.linalg.inv(K),(img_p)*d_t))
                    res[i,0:3] = aa
    return res

#####################
#####################
# fcns for segmentation

def cut_save_seg(maskfile):
    masks = glob.glob(maskfile)
    for maskf in masks:
        mask = cv2.imread(maskf)
        newHeight = 128
        newWidth = 160
        mask = cv2.resize(mask[15:384,46:508,:], (newWidth, newHeight))
        cv2.imwrite(maskf, mask)
    return 0


def get_segmentation(imagefile, recalcul=True):
    old_images = glob.glob(seg_image_file+'/*.jpg')
    for file in old_images:
        os.remove(file)

    old_images = glob.glob(seg_final+'/*.png')
    for file in old_images:
        os.remove(file)

    t = 1
    for file in imagefile:
        im = cv2.imread(file)
        Height = 128*3
        Width = 160*3
        im = cv2.resize(im, (Width, Height))
        cv2.imwrite(seg_image_file + '/' + str(t) + '.jpg', im)
        t = t + 1
    with open(seg_image_list, "w") as text_file:
        for i in range(t-1):
            print('{:d}.jpg'.format(i+1), file=text_file)

    if recalcul:
        old_segs = glob.glob(seg_out+'/*.png')
        for file in old_segs:
            os.remove(file)
        os.system('python '+ seg_script)

    for i in range(t-1):
        file = seg_out+'/' +str(i+1)+ '_mask.png'
        im = cv2.imread(file)
        im= cv2.resize(im[15:384,46:508,:], (newWidth, newHeight))
        cv2.imwrite(seg_final+'/'+str(i+1)+'.png', im)
    return 0

def seperate_seg():
    segs = glob.glob(seg_final+'/*.png')
    for i in range(len(segs)):
        file = seg_final+'/' +str(i+1)+ '.png'
        im = cv2.imread(file)
        seperate_seg_single(im, i+1)
    return 0

def seperate_seg_single(im, iii):
    im = cv2.resize(im, (newWidth, newHeight))

    # get the class labels
    a, b = np.histogram(im.ravel(), bins=100, range=(0.1, 255.0))
    maxclass = 10
    label = list()
    for i in range(maxclass):
        idx = np.argmax(a)
        if a[idx] > 50:
            a[idx] = 0
            label.append(b[idx])
    #seperate the segmentation

    seg_save = 'keys/seg/'
    for k in range(len(label)):
        res = np.zeros((newHeight,newWidth))
        for i in range(newHeight):
            for j in range(newWidth):
                if (label[k]-1 <= im[i,j,0] <= label[k]+1 or
                label[k]-1 <= im[i,j,1] <= label[k]+1 or label[k]-1 <= im[i,j,2] <= label[k]+1) :
                    res[i,j] = int(label[k])
        if not os.path.exists(seg_save + str(iii)):
            os.makedirs(seg_save + str(iii))
        seperate_objs(res,seg_save , int(label[k]) , iii)

def seperate_objs(seg, seg_save , k, iii):
    ss = cv2.ximgproc.segmentation.createGraphSegmentation(0.5,300,450)
    seg1 = ss.processImage(seg)
    labels = np.unique(seg1)
    #print(labels)
    for i in range(len(labels)-1):
        lab = labels[i+1]
        mask1 = np.ma.masked_equal(seg1, lab)
        out_file = seg_save + str(iii) + '/'+ str(k)+ '_' + str(i) +'.npy'
        np.save(out_file, mask1.mask)
    return 0
#####################
#####################
# fcns for bounding box

def draw_floor(x = 0.95, n=201):
    res = np.zeros((n*n ,6),dtype = np.float32)+255
    res[:, 0] = x
    desy = 2.5/n
    desz = (4.3-1)/n
    for i in range(n):
        for j in range(n):
            res[i*n+j, 1] =  desy*i
            res[i*n+j, 2] =  -0.8+desz*j
    return res

def find_bounding_box(res):
    #find bounding box from point cloud
    x_max = res[:,0].max()
    x_min = res[:,0].min()
    y_max = res[:,1].max()
    y_min = res[:,1].min()
    z_max = res[:,2].max()
    z_min = res[:,2].min()
    #bb = draw_box(x_max,x_min, y_max,y_min,z_max,z_min,'0_255_0')
    bb = 0
    return bb, np.array([x_max,x_min, y_max,y_min,z_max,z_min])


def draw_polygen(top,z_max,z_min, off_set,  color):
    res = np.zeros((1,6))
    for i in range(len(top)):
        j = 0 if i+1==len(top) else i+1
        res1 = draw_line(np.array([z_min,top[i][0]+off_set[0],top[i][1]+off_set[1]]),np.array([z_min,top[j][0]+off_set[0],top[j][1]+off_set[1]]),color)
        res2 = draw_line(np.array([z_max,top[i][0],top[i][1]]),np.array([z_max,top[j][0],top[j][1]]),color)
        res3 = draw_line(np.array([z_min,top[i][0]+off_set[0],top[i][1]+off_set[1]]),np.array([z_max,top[i][0],top[i][1]]),color)
        res = np.concatenate((res,res1,res2,res3), axis=0)
    return res[1:]

def draw_box(x_max,x_min, y_max,y_min,z_max,z_min, color):
    res1 = draw_line(np.array([x_min,y_min,z_min]),np.array([x_min,y_min,z_max]),color)
    res2 = draw_line(np.array([x_min,y_min,z_min]),np.array([x_min,y_max,z_min]), color)
    res3 = draw_line(np.array([x_min,y_min,z_min]),np.array([x_max,y_min,z_min]), color)
    res4 = draw_line(np.array([x_min,y_min,z_max]),np.array([x_min,y_max,z_max]), color)
    res5 = draw_line(np.array([x_min,y_min,z_max]),np.array([x_max,y_min,z_max]), color)
    res6 = draw_line(np.array([x_max,y_min,z_max]),np.array([x_max,y_min,z_min]), color)
    res7 = draw_line(np.array([x_min,y_max,z_max]),np.array([x_min,y_max,z_min]), color)
    res8 = draw_line(np.array([x_max,y_max,z_max]),np.array([x_max,y_max,z_min]), color)
    res9 = draw_line(np.array([x_max,y_max,z_max]),np.array([x_max,y_min,z_max]), color)
    res10 = draw_line(np.array([x_max,y_max,z_max]),np.array([x_min,y_max,z_max]), color)
    res11 = draw_line(np.array([x_max,y_max,z_min]),np.array([x_max,y_min,z_min]), color)
    res12 = draw_line(np.array([x_max,y_max,z_min]),np.array([x_min,y_max,z_min]), color)
    bb = np.concatenate((res1,res2,res3,res4,res5,res6,res7,res8,res9,res10,res11,res12), axis=0)
    return bb


def draw_line(x,y, color):
    x = x.reshape(1,3)
    y = y.reshape(1,3)
    n = sqrt( (x[0,0] - y[0,0])**2 + (x[0,1] - y[0,1])**2+(x[0,2] - y[0,2])**2 )/0.01
    n = max(int(n),2)
    res = np.zeros((n ,6),dtype = np.float32)
    des = (y - x)/(n-1)
    for i in range(n):
        res[i, 0:3] = x + des*i
        if color == 'camera':
            res[i, 3:6] = np.array([0,255,255])
        else:
            tt = color.split("_")
            res[i, 3] = int(tt[0])
            res[i, 4] = int(tt[1])
            res[i, 5] = int(tt[2])
    return res

def get_object_bb(imagefile, dep_masked,T):
    image = cv2.imread(imagefile)
    image = cv2.resize(image, (newWidth, newHeight))
    res = vert(image, dep_masked.reshape((newHeight ,newWidth)), T,  False)
    if len(res) == 0:
        return None,None
    else:
        #res = refine_seg_3D(res)
        bbres, bb = find_bounding_box(res)
        return bbres,bb

def update_bb(bb1,bb2):
    if bb2 is None:
        return bb1
    if bb1 is None:
        return bb2
    bb1[0] = np.min((bb1[0],bb2[0]))
    bb1[1] = np.max((bb1[1],bb2[1]))
    bb1[2] = np.min((bb1[2],bb2[2]))
    bb1[3] = np.max((bb1[3],bb2[3]))
    bb1[4] = np.min((bb1[4],bb2[4]))
    bb1[5] = np.max((bb1[5],bb2[5]))
    return bb1

def if_pt_in_bb(pt, bb):
    if bb[0]>pt[0]>bb[1] and bb[2]>pt[1]>bb[3] and bb[4]>pt[2]>bb[5]:
        return True
    else:
        return False

def delete_pts_in_bb(res, bb):
    res0 = np.zeros((len(res),6),dtype = np.float32)
    t = 0
    for i in range(len(res)):
        pt = np.array([res[i,0],res[i,1],res[i,2]])
        if not if_pt_in_bb(pt, bb):
            res0[t,:] = res[i,:]
            t = t + 1
    return res0[0:t]



def add_image_bb(imagefile, depfile):
    print('Start matching the image ', imagefile)
    image0 = cv2.imread(imagefile)
    rr = 2
    image = cv2.resize(image0, (newWidth*rr, newHeight*rr))
    dep = np.load(depfile)

    images = glob.glob('keys/*.jpg')
    Ms = []
    l = len(images)
    for i in range(max(1,l-3), l+1):
    #for i in range(1, min(l+1,2)):
        img0 = cv2.imread( 'keys/'+str(i)+'.jpg')
        img0 = cv2.resize(img0, (newWidth*rr, newHeight*rr))
        pts0, pts, kp1, kp2, matches, matchesMask = sift_points(img0,image,0.65)
        d0 = np.load('keys/deps/'+str(i)+'.npy')
        #dep, _ = corres_imgs(image,img0,dep,d0)
        print('   matching the image with image '+ str(i))
        print('      -> the translation matrix is :')
        if len(pts) != 0:
            pts0 = verttt(pts0,d0[0,:,:,0] ,rr)
            pts = verttt(pts,dep[0,:,:,0] ,rr)
            retval, M, inliers = cv2.estimateAffine3D(pts, pts0, ransacThreshold=0.8)
            #M = adjust_m(M)
        else :
            M = None

        if M is not None:
            print('      ->   ', M[0])
            print('      ->   ', M[1])
            print('      ->   ', M[2])
            M0 = np.load('keys/Ms/'+str(i)+'.npy')
            Ms.append(mut_trans(M,M0))
        else:
            print('      ->   None')
    if l == 0:
        Ms.append(np.eye(3,4))
    image = cv2.resize(image0, (newWidth, newHeight))

    if len(Ms) != 0:
        #M = Ms[len(Ms)-1]
        M = average_matrix_v3(Ms)
        masksfile = glob.glob('keys/seg/'+str(l+1)+'/*.npy')
        for mask_file in masksfile:
            tt = len(mask_file)
            label = mask_file[tt-8:tt-4]
            mask = np.load(mask_file)
            dep_masked = dep.reshape((newHeight ,newWidth))*(mask*1)
            dep = dep.reshape((newHeight ,newWidth)) - dep_masked
            _, bb = get_object_bb(imagefile, dep_masked,M)
            res0 = vert(image, dep, M, False)

            if os.path.isfile('keys/objs/' + label + '.npy'):
                bb = update_bb(bb, np.load('keys/objs/' + label + '.npy'))

            resbb = draw_box(bb[0],bb[1],bb[2],bb[3],bb[4],bb[5],color=label[0:2])
            write_ply('keys/res/bb'+ label +'.ply', resbb)

            res_obj = vert(image, dep_masked, M, False)
            write_ply('keys/objs/'+ str(l+1) + label+'.ply', res_obj)
            #res0 = delete_pts_in_bb(res0, bb)
            np.save('keys/objs/'+ label +'.npy' , bb)

        write_ply('keys/res/'+str(l+1) +'.ply', res0)

        np.save('keys/Ms/'+str(l+1) +'.npy' , M)
        cv2.imwrite('keys/'+str(l+1) +'.jpg',image0)
        print(' the process of image '+ str(l+1) + ' is finished.')
    else:
        print(' cannot match image '+ imagefile +  '.')
    print(' ')
    return 0


def add_image_bb_v2(imagefile, depfile, old_res):
    print('Start matching the image ', imagefile)
    image0 = cv2.imread(imagefile)
    rr = 1
    image = cv2.resize(image0, (newWidth*rr, newHeight*rr))
    dep = np.load(depfile)

    mtx = np.eye(3)
    mtx[0,0]=200
    mtx[1,1]=200
    dst = np.zeros((1,5))

    images = glob.glob('keys/*.jpg')
    Ms = []
    l = len(images)
    for i in range(max(1,l-3), l+1):
    #for i in range(1, min(l+1,2)):
        img0 = cv2.imread( 'keys/'+str(i)+'.jpg')
        img0 = cv2.resize(img0, (newWidth*rr, newHeight*rr))
        pts0, pts, kp1, kp2, matches, matchesMask = sift_points(img0,image,0.5)
        d0 = np.load('keys/deps/'+str(i)+'.npy')
        #dep, _ = corres_imgs(image,img0,dep,d0)
        print('   matching the image with image '+ str(i))
        print('      -> the translation matrix is :')
        if len(pts) != 0:
            pts0 = verttt(pts0,d0[0,:,:,0] ,rr)
            pts = verttt(pts,dep[0,:,:,0] ,rr)
            #retval, M, inliers = cv2.estimateAffine3D(pts, pts0, ransacThreshold=0.8)
            M = rigid_transform_3D(pts.reshape(-1,3), pts0.reshape(-1,3))
            #M = adjust_m(M)
        else :
            M = None

        if M is not None:
            print('      ->   ', M[0])
            print('      ->   ', M[1])
            print('      ->   ', M[2])
            M0 = np.load('keys/Ms/'+str(i)+'.npy')
            Ms.append(mut_trans(M,M0))
        else:
            print('      ->   None')
    if l == 0:
        Ms.append(np.eye(3,4))
    image = cv2.resize(image0, (newWidth, newHeight))

    dep = erosion_seg(image, dep)
    # "image" is the new added image "dep" is its depth map
    if len(Ms) != 0:
        M = Ms[len(Ms)-1]
        #M = average_matrix_v3(Ms)
        if old_res is None:
            res_final = vert(image, dep.reshape((newHeight,newWidth)), M, False)
        else:
            imgr, depr, res_out = projection_3d_2d_v1(old_res, M, mtx)
            new_dep = combine_dep_known_rgb(image, dep.reshape((newHeight,newWidth)), imgr, depr)
            res_new_image = vert(image, new_dep.reshape((newHeight,newWidth)), M, False)
            res_final = np.concatenate((res_out,res_new_image), axis=0)

        res_new = vert(image, dep.reshape((newHeight,newWidth)), M, False)
        write_ply('keys/res/nnnn'+str(l+1) +'.ply', res_new)
        write_ply('keys/res/'+str(l+1) +'.ply', res_final)

        np.save('keys/Ms/'+str(l+1) +'.npy' , M)
        cv2.imwrite('keys/'+str(l+1) +'.jpg',image0)
        print(' the process of image '+ str(l+1) + ' is finished.')
    else:
        res_final = old_res
        print(' cannot match image '+ imagefile +  '.')
    print(' ')
    return res_final
