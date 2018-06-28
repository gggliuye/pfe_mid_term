import sys
import cv2
import numpy as np
import glob
import os
from utils import *
from utils_refine import *
from math import sqrt
from numpy import *
from shapely.geometry import Polygon
import pickle
from scipy.io import loadmat

index_file = 'keys/objs/index.npy'

### floor and camera

def draw_camera(T,inv = False):
    T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)
    if inv:
        T = np.linalg.inv(T)
    T = T[0:3,:]
    pos = []
    s = 0.06
    pos.append([0,0,1.5*s,1])
    pos.append([s,s,0,1])
    pos.append([s,-s,0,1])
    pos.append([-s,-s,0,1])
    pos.append([-s,s,0,1])

    pos = np.array(pos)
    n_cam = (np.dot(T,pos.T)).T

    res = np.zeros((1,6))
    for i in range(1,5):
        res1 = draw_line(n_cam[0],n_cam[i], 'camera')
        j = 1 if i==4 else i+1
        res2 = draw_line(n_cam[i],n_cam[j], 'camera')
        res = np.concatenate((res,res1,res2), axis=0)
    return res[1:]

def detect_floor(res):
    a, b = np.histogram(res[:, 0].ravel(), bins=100, range=(0.1, 2.0))
    idx = np.argmax(a)
    out = b[idx]
    np.save('keys/objs/floor'+str(int(out*100)) +'.npy' , out)
    return out

def refine_result_with_floor(res):
    x_floor = detect_floor(res)
    #delete pts around the floor
    t = 0
    for i in range(len(res)):
        if not x_floor-0.04 < res[i,0] < x_floor+0.19:
            res[t, :] = res[i, :]
            t = t + 1
    res_floor = draw_floor(x_floor)
    return np.concatenate((res[0:t, :], res_floor), axis=0)

def transform_vert_nonfloor(res,T):
    x_floor = detect_floor(res)
    out = np.zeros((len(res),6))
    t = 0
    for i in range(len(res)):
        if not x_floor-0.04 < res[i,0] < x_floor+0.19:
            real_p = res[i,0:3]
            real_p = np.append(real_p, 1)
            aa = np.dot(T, real_p).reshape(1,3)
            out[t,0:3] = aa[0,0:3]
            out[t,3:6] = res[i,3:6]
            t = t + 1
    return res[0:t, :]

##########
## fcns for ransac rigid_transform_3D
def take_idx_value(A, idx):
    ran = np.zeros((len(idx), 3))
    for i in range(len(idx)):
        ran[i,:] = A[idx[i],:]
    return ran

def count_inlier(A,B,T,thresh):
    count = 0
    idx = []
    for i in range(len(A)):
        err = np.dot(T[0:3,0:3],A[i].T) + T[:,3] - B[i].T
        if np.sum(np.abs(err)) < thresh:
            count = count + 1
            idx.append(i)
    return count, idx

def rigid_transform_3D_ransac(A, B, thresh=0.3 ,k = 200):
    N = len(A)
    num = 10

    thresh_num_inlier = 0.9*N
    max_inlier = 0.5*N
    max_inlier_idx = 0
    for i in range(k):
        idx = np.random.randint(N, size = num)
        ransca_a = take_idx_value(A, idx)
        ransca_b = take_idx_value(B, idx)
        T = rigid_transform_3D(ransca_a,ransca_b)
        inlier ,idx_inlier = count_inlier(A,B,T,thresh)
        if inlier > thresh_num_inlier:
            ransca_a = take_idx_value(A, idx_inlier)
            ransca_b = take_idx_value(B, idx_inlier)
            T = rigid_transform_3D(ransca_a,ransca_b)
            return T
        if inlier > max_inlier:
            max_inlier_idx = idx_inlier
            max_inlier = inlier

    if max_inlier_idx == 0:
        return None
    else:
        ransca_a = take_idx_value(A, max_inlier_idx)
        ransca_b = take_idx_value(B, max_inlier_idx)
        T = rigid_transform_3D(ransca_a,ransca_b)
        return T

################
################
## fcns for boundingbox

def draw_boundbox(index_fil = index_file):
    t = 0
    res = np.zeros((1,6))
    with open(index_fil, 'rb') as f:
        index = pickle.load(f)
    for idx in index:
        top = idx['top_coord']
        z_max = idx["tall"][0]
        z_min = idx["tall"][1]
        label = idx["label"]
        off_set = idx["offset"]
        color = np.random.randint(100)
        res1 = draw_polygen(top,z_max,z_min,off_set,color)
        t = t + 1
        res = np.concatenate((res,res1), axis=0)
    write_ply('keys/res/bb_'+ str(t) +'.ply', res[1:,:])
    return 0


def if_same_object(bb, bb_new):
    #if bb["label"] != bb_new["label"]:
    #    return False
    if np.linalg.norm(bb["tall"] - bb_new["tall"]) > 0.3:
        return False
    cg0 = np.mean(np.array(bb["top_coord"]), axis=0) + bb["offset"]/2
    cg_new0 = np.mean(np.array(bb_new["top_coord"]), axis=0) + bb_new["offset"]/2
    if np.linalg.norm(cg0 - cg_new0) > 0.3:
        return False

    cg = np.array([cg0[0], cg0[1], np.mean(bb["tall"])])
    cg_new = np.array([cg_new0[0], cg_new0[1], np.mean(bb_new["tall"])])

    seuil = np.mean(np.linalg.norm(bb["top_coord"], axis = 1))/4
    if np.linalg.norm(cg-cg_new) >  seuil:
        return False
    return True

def projection_idx(idxold,T, tall):
    T = np.array(T)
    pt1_prej = []
    pt2_prej = []
    length = len(idxold)
    for i in range(length):
        pt1 = np.array([tall[0],idxold[i][0], idxold[i][1] ])
        pt2 = np.array([tall[1],idxold[i][0], idxold[i][1] ])
        pt1_prej.append(np.dot(T[0:3,0:3], pt1.T).reshape(3) + T[0:3, 3].reshape(3))
        pt2_prej.append(np.dot(T[0:3,0:3], pt2.T).reshape(3) + T[0:3, 3].reshape(3))
    pt1_prej = np.array(pt1_prej)
    pt2_prej = np.array(pt2_prej)
    #temp = (np.linalg.norm(pt1_prej[0,1:3]-pt2_prej[0,1:3]))/tall[0]-tall[1]
    #alpha = arcsin(temp)
    #calcule tall
    tall[0] = np.max(pt1_prej[:,0])
    tall[1] = np.min(pt2_prej[:,0])

    idx = []

    for i in range(length):
        t = (tall[0]-pt2_prej[i,0])/(pt2_prej[i,0]-pt1_prej[i,0])
        t0 = pt2_prej[i,1] + t*(pt2_prej[i,1]-pt1_prej[i,1])
        t1 = pt2_prej[i,2] + t*(pt2_prej[i,2]-pt1_prej[i,2])
        idx.append((t0,t1))


    offset = np.zeros(2)
    t = (tall[1]-pt2_prej[0,0])/(pt2_prej[0,0]-pt1_prej[0,0])
    offset[0] = pt2_prej[0,1] + t*(pt2_prej[0,1]-pt1_prej[0,1]) - np.float(idx[0][0])
    offset[1] = pt2_prej[0,2] + t*(pt2_prej[0,2]-pt1_prej[0,2]) - np.float(idx[0][1])

    return idx, offset, tall

def choose_offset(os1, os2):
    if np.linalg.norm(os1) < np.linalg.norm(os2):
        return os1
    else:
        return os2

def update_bb_v2(index,bb, T, label):
    if bb is None:
        return index
    bb_new = {}
    idx = []
    idx.append((bb[3], bb[5]))
    idx.append((bb[2], bb[5]))
    idx.append((bb[2], bb[4]))
    idx.append((bb[3], bb[4]))
    idx, offset, tall = projection_idx(idx, T, bb[0:2])
    bb_new["top_coord"] = idx
    bb_new["tall"] = tall
    bb_new["label"] = label
    bb_new["offset"] = offset
    if index is None:
        index_new = []
        index_new.append(bb_new)
        return index_new

    p_new = Polygon(bb_new["top_coord"])

    for i in range(len(index)):
        p_old = Polygon(index[i]["top_coord"])
        ppinter = p_old.intersection(p_new)
        if if_same_object(index[i], bb_new) and (not ppinter.is_empty):
            index[i]["top_coord"] = list(ppinter.exterior.coords)
            index[i]["tall"] = (bb_new["tall"] + index[i]["tall"])/2
            index[i]["offset"] = choose_offset(bb_new["offset"], index[i]["offset"])
            return index

    index.append(bb_new)
    return index

def get_object_bb_new(imagefile, dep_masked, mtx):
    image = cv2.imread(imagefile)
    image = cv2.resize(image, (newWidth, newHeight))
    res = vert_new(image, dep_masked.reshape((newHeight,newWidth)),  np.eye(3,4),mtx, False)
    if len(res) == 0:
        return None,None
    else:
        bbres, bb = find_bounding_box(res)
        return bbres,bb

def make_bounding_box(M,l,dep,imagefile, mtx):
    masksfile = glob.glob('keys/seg/'+str(l+1)+'/*.npy')
    for mask_file in masksfile:
        tt = len(mask_file)
        label = mask_file[tt-8:tt-4]
        mask = np.load(mask_file)
        dep_masked = dep.reshape((newHeight ,newWidth))*(mask*1)
        dep_non_obj = dep.reshape((newHeight ,newWidth)) - dep_masked
        _, bb = get_object_bb_new(imagefile, dep_masked, mtx)

        if os.path.isfile(index_file):
            with open(index_file, 'rb') as f:
                index = pickle.load(f)
            index = update_bb_v2(index, bb, M, label)
            with open(index_file, 'wb') as f:
                pickle.dump(index, f)
        else:
            bb,index = update_bb_v2(None, bb,M, label)
            with open(index_file, 'wb') as f:
                pickle.dump(index, f)

        #save the point cloud of the masked object
        #res_obj = vert(image, dep_masked, M, False)
        #write_ply('keys/objs/'+ str(l+1) + label+'.ply', res_obj)

        #delete the points in the bounding box
        #res0 = delete_pts_in_bb(res0, bb)
        np.save('keys/objs/'+ label +'.npy' , bb)
        return 0

##### calculate transformation matrix for different situations
#####
def calcul_M(image0, dep, mtx, dst):
    rr = 1
    image = cv2.resize(image0, (newWidth*rr, newHeight*rr))

    images = glob.glob('keys/images/img/*.jpg')
    Ms = []
    l = len(images)
    for i in range(max(1,l-2), l+1):
    #for i in range(1, min(l+1,2)):
        img0 = cv2.imread( 'keys/images/img/'+str(i)+'.jpg')
        img0 = cv2.resize(img0, (newWidth*rr, newHeight*rr))
        pts0, pts, kp1, kp2, matches, matchesMask = sift_points(img0,image,0.55)
        d0 = np.load('keys/deps/dep_'+str(i)+'.npy')
        #dep, _ = corres_imgs(image,img0,dep,d0)
        print('   matching the image with image '+ str(i))
        print('      -> the translation matrix is :')
        if len(pts) != 0:
            #pts0 = verttt(pts0,d0[0,:,:,0] ,rr)
            #pts = verttt(pts,dep[0,:,:,0] ,rr)
            pts0 = pts2d_to_pts3d(pts0,d0[0,:,:,0],mtx)
            pts = pts2d_to_pts3d(pts,dep[0,:,:,0],mtx)
            #retval, M, inliers = cv2.estimateAffine3D(pts, pts0, ransacThreshold=0.8)
            M = rigid_transform_3D_ransac(pts.reshape(-1,3), pts0.reshape(-1,3),0.5)
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

    return Ms,l

def calcul_M_round(image0, dep, mtx, dst, num=10):
    Ms = []
    images = glob.glob('keys/images/img/*.jpg')
    l = len(images)
    i = l+1

    angle = np.pi*l*(1.6)/(num)
    r = np.array([angle,0,0])
    t = np.zeros(3)
    M = trans_matrix(r,t)
    print('      ->   ', M[0])
    print('      ->   ', M[1])
    print('      ->   ', M[2])
    Ms.append(M)
    return Ms, l

def calcul_M_ekfslam():
    Ms = []
    M = loadmat('F:/pfe/kalman_filter/ekfslam/matlab_code/camera_matrix.mat');
    images = glob.glob('keys/images/img/*.jpg')
    l = len(images)
    if l == 0 :
        M = np.eye(3,4)
    else:
        temp = M['camera_matrix'][l-1]
        M = np.zeros((3,4))
        M[:,3] = temp[0:3]
        M[:,0:3] = temp[3:12].reshape(3,3)

    print('      ->   ', M[0])
    print('      ->   ', M[1])
    print('      ->   ', M[2])
    Ms.append(M)
    return Ms, l

def q2r(q):
    [x,y,z,r] = [q[0], q[1], q[2], q[3]]
    rr = []
    rr.append(np.array([r*r+x*x-y*y-z*z, 2*(x*y-r*z), 2*(x*z+r*y)]))
    rr.append(np.array([2*(x*y+r*z), r*r-x*x+y*y-z*z, 2*(y*z-r*x)]))
    rr.append(np.array([2*(z*x-r*y), 2*(y*z+r*x), r*r-x*x-y*y+z*z]))
    return np.asarray(rr)

def calcul_M_orbslam2():
    Ms = []
    M = np.loadtxt('F:/pfe/slam/ORB_slam2/KeyFrameTrajectory.txt')
    images = glob.glob('keys/images/img/*.jpg')
    l = len(images)
    temp = M[l]
    M = np.zeros((3,4))
    M[0,3] = temp[2]
    M[1,3] = temp[3]
    M[2,3] = temp[1]

    R = q2r(temp[4:8])
    R_angle = cv2.Rodrigues(R)[0]
    R_angle[0] = R_angle[1]
    R_angle[1] = -R_angle[2]
    R_angle[2] = R_angle[0]/3
    M[:,0:3] = cv2.Rodrigues(R_angle)[0]

    print('      ->   ', M[0])
    print('      ->   ', M[1])
    print('      ->   ', M[2])
    Ms.append(M)
    return Ms, l

####### core functions
#######

def add_image_bb_com(imagefile, depfile, old_res, seg = False):
    print('Start matching the image ', imagefile)
    image0 = cv2.imread(imagefile)
    dep = np.load(depfile)

    mtx = np.eye(3)
    mtx[0,0]=517.306408/4
    mtx[1,1]=516.469215/4
    mtx[0,2]=80
    mtx[1,2]=64
    dst = np.zeros((1,5))

    #Ms,l = calcul_M(image0, dep, mtx, dst)

    #Ms,l = calcul_M_round(image0, dep, mtx, dst)

    #Ms,l = calcul_M_ekfslam()

    Ms, l = calcul_M_orbslam2()

    image = cv2.resize(image0, (newWidth, newHeight))

    # "image" is the new added image "dep" is its depth map
    if len(Ms) != 0:
        M = Ms[len(Ms)-1]
        #M = average_matrix_v3(Ms)
        if seg:
            make_bounding_box(M,l,dep,imagefile, mtx)

        #dep = erosion_seg(image, dep)

        if old_res is None:
            res_final = vert_new(image, dep.reshape((newHeight,newWidth)), M,mtx, True)
            #res_final = refine_result_with_floor(res_final)
        else:
            #imgr, depr, res_out = projection_3d_2d_v1(old_res, M, mtx)
            #new_dep = combine_dep_known_rgb(image, dep.reshape((newHeight,newWidth)), imgr, depr)
            new_dep = dep
            res_new_image = vert_new(image, new_dep.reshape((newHeight,newWidth)), M,mtx, True)
            res_final = np.concatenate((old_res,res_new_image), axis=0)
            #res_new_image = transform_vert_nonfloor(res_new_image,M)

            #res_final = np.concatenate((res_out,res_new_image), axis=0)
        res_camera = draw_camera(M, True)
        write_ply('keys/cam/' +str(l+1) +'.ply', res_camera)
        write_ply('keys/res/'+str(l+1) +'.ply', res_final)
        np.save('keys/Ms/'+str(l+1) +'.npy' , M)
        cv2.imwrite('keys/images/img/'+str(l+1) +'.jpg',image0)
        print(' the process of image '+ str(l+1) + ' is finished.')
    else:
        res_final = old_res
        print(' cannot match image '+ imagefile +  '.')
    print(' ')
    return res_final
