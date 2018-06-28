import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import os
from pathlib import Path

# global variables
newHeight = 128
newWidth = 160
ckptfile = 'F:/pfe/cnn_slam/FCRN-DepthPrediction-master/ckpt/NYU_FCRN.ckpt'


def reform_pts(pts):
    out = np.zeros((len(pts),1,2), dtype = np.float32)
    for i in range(len(pts)):
        out[i,0,:] = pts[i,:]
    return out

def get_camera_matrix(imagesfile):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    images = glob.glob(imagesfile)
    objp = np.zeros((5*5,3), np.float32)
    objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        img = cv2.imread(fname)
        img = cv2.resize(img, (newWidth, newHeight))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (5,5),None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    #print(rvecs, tvecs)
    return mtx, dist,imgpoints

def trans_matrix(r,t):
    # T (ki to t), rotate and transport matrix
    r = r.ravel()
    t = t.ravel()
    (a,b,c) = (r[0], r[1], r[2])
    (x,y,z) = (t[0], t[1], t[2])
    Rz = np.matrix([[1,0,0],
          [0, np.cos(a), -np.sin(a)],
         [0, np.sin(a), np.cos(a)]])
    Ry = np.matrix([[np.cos(b), 0 , np.sin(b)],
          [0 , 1, 0],
          [-np.sin(b), 0 , np.cos(b)]])
    Rx = np.matrix([[np.cos(c), -np.sin(c), 0],
         [np.sin(c), np.cos(c), 0],
         [0,0,1]])
    R = np.dot(Rz,np.dot(Ry,Rx))
    T = np.concatenate((R, np.matrix([[x],[y],[z]])), axis=1)

    return T


def transpose_matrix(fname, mtx, dist):
    objp = np.zeros((5*5,3), np.float32)
    objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (5,5),None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
    else:
        return False
    T = trans_matrix(rvecs, tvecs)

    return T

def combine(dep1, dep2):
    # average dep1 and dep2 (for value non zero)
    dep1 = dep1.reshape((newHeight,newWidth))
    dep2 = dep2.reshape((newHeight,newWidth))
    for i in range(newHeight):
        for j in range(newWidth):
            if dep1[i,j] != 0 and dep2[i,j] != 0:
                dep1[i,j] = (dep1[i,j]+dep2[i,j])/2
    return dep1.reshape((1,newHeight,newWidth,1))

def corres_imgs(im1,im2,dep1,dep2):
    #im1 is the original image
    #match im2 to im1
    #match points to find M
    pts1, pts2, kp1, kp2, matches, matchesMask = sift_points(im1,im2,0.75)
    if len(pts1) > 3:
        src_pts = np.float32(pts1).reshape(-1,1,2)
        dst_pts = np.float32(pts2).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,12)
        dst1 = cv2.warpPerspective(dep1[0,:,:,0],(M),(newWidth,newHeight))
        dst2 = cv2.warpPerspective(dep2[0,:,:,0],np.linalg.inv(M),(newWidth,newHeight))
        dep1 = combine(dep1,dst2)
        dep2 = combine(dep2,dst1)

    return dep1, dep2


def vert(img,D, T, inv = True, foc = 200):
    l = img.shape[0]
    h = img.shape[1]
    t = 0
    temp = np.zeros((l*h, 6))
    T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)
    if inv:
        T = np.linalg.inv(T)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 0.1<D[i,j]< 200000:
                ddd = D[i,j]
                u = np.array([i*ddd/foc,j*ddd/foc,ddd, 1])
                aa = np.dot(T, u)
                x = np.zeros(3)
                aa = aa.reshape((1,4))
                x[0:3] = aa[0,0:3]/aa[0,3]
                #if tempp :
                #    x[0:3] = aa[0,0:3]/aa[0,3]
                #else:
                #    x[0:3] = aa[0:3]/aa[3]
                temp[t,0] = x[0]
                temp[t,1] = x[1]
                temp[t,2] = x[2]
                temp[t,3:6] = img[i,j]
                t = t+1
    return temp[0:t,:]


def pts2d_to_pts3d(pts,D,K):
    # projection from 2d points top 3d points in camera world
    res = np.zeros((1,len(pts) ,3),dtype = np.float32)
    for i in range(len(pts)):
        dep = D[int(pts[i][1]),int(pts[i][0])]
        img_p = np.array([pts[i][1], pts[i][0], 1])
        res[0][i][0:3] = np.dot(np.linalg.inv(K),(img_p)*dep)
    return res

def vert_new(img,D, T,K, inv = True):
    l = img.shape[0]
    h = img.shape[1]
    t = 0
    temp = np.zeros((l*h, 6))
    T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)
    if inv:
        T = np.linalg.inv(T)

    for i in range(l):
        for j in range(h):
            dep = D[i,j]
            if 0.1<dep< 200000:
                #camera caliberation
                img_p = np.array([i, j, 1])
                real_p = np.dot(np.linalg.inv(K),(img_p)*dep)
                real_p = np.append(real_p, 1)
                #real_p[1] = - real_p[1]
                aa = np.dot(T, real_p).reshape(1,4)
                temp[t,0:3]  = aa[0,0:3]/aa[0,3]
                temp[t,3:6] = img[i,j]
                t = t+1
    return temp[0:t,:]


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
end_header
'''
def write_ply(fn,verts ):
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')


def verttt(pts,D,ratio =1, foc = 200):
    # old version of projection from 2d points to 3d points
    # bad camera caliberation in this fcn
    res = np.zeros((1,len(pts) ,3),dtype = np.float32)
    for i in range(len(pts)):
        res[0][i][0] = int(pts[i][1]/ratio)
        res[0][i][1] = int(pts[i][0]/ratio)
        res[0][i][2] = D[int(pts[i][1]/ratio),int(pts[i][0]/ratio)]
        res[0][i][0] = res[0][i][0]*res[0][i][2]/foc
        res[0][i][1] = res[0][i][1]*res[0][i][2]/foc
    return res

def mask_img(image, mask):
    for i in range(3):
        image[:,:,i] = image[:,:,i]*mask
    return image

def erosion_seg(img, dep):
    dep = dep.reshape((newHeight ,newWidth))
    ss = cv2.ximgproc.segmentation.createGraphSegmentation(0.5,100,100)
    seg = ss.processImage(img)
    temp = np.unique(seg)
    result = np.zeros(dep.shape, dtype=np.float32)
    kernel = np.ones((3,3),np.uint8)
    for i in range(len(temp)):
        mask0 = np.ma.masked_equal(seg, temp[i])
        erosion = cv2.erode(dep*(mask0.mask),kernel,iterations = 1)
        #erosion = cv2.dilate(erosion,kernel,iterations = 1)
        result = result + erosion
        #result = cv2.dilate(result,kernel,iterations = 1)
    return result


def del_point(depth, mask):
    depth = depth*mask
    a,b = np.histogram(depth.ravel(), bins=100, range=(0.1, 5.0))
    seuil = b[np.argmax(a)]
    mm1 = np.ma.masked_where(depth<seuil+0.2,depth)
    mm2 = np.ma.masked_where(depth>seuil-0.2,depth)
    return depth*mm1.mask*mm2.mask

def del_all(image, depth):
    ss = cv2.ximgproc.segmentation.createGraphSegmentation(0.5,100,100)
    seg = ss.processImage(image)
    temp = np.unique(seg)
    result = np.zeros(depth.shape, dtype=np.float32)
    for i in range(len(temp)):
        mask0 = np.ma.masked_equal(seg, temp[i])
        d_t = del_point(depth, mask0.mask)
        result = result + d_t
    return result

def redes(kp, des, sss = 128):
    # add height information to 'des'
    # add width information to 'des'
    temp1 = 1.5
    temp2 = 5
    h, w = des.shape
    res = np.zeros((h,w+2), dtype= np.float32)
    for i in range(h):
        res[i,0:w] = des[i,:]
        res[i,w] = kp[i].pt[1]/(sss*temp1)
        res[i,w+1] = kp[i].pt[0]/(sss*temp2)
    return res

def sift_points(im1,im2,param =0.55):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1,None)
    des1 = redes(kp1,des1)
    kp2, des2 = sift.detectAndCompute(im2,None)
    des2 = redes(kp2,des2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []
    good = []
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < param*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i]=[1,0]
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2, kp1,kp2, matches, matchesMask

def feature_points(im1, dep, mtx):
    #input image and Depth, output 3d feature points and their descriptors

    dep = dep.reshape((newHeight,newWidth))
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SURF_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1,None)
    des1 = redes(kp1,des1)
    kp = np.zeros((len(kp1), 3))


    ## use 2d-3d trans and mtx
    for i in range(len(kp1)):
        kp[i,0] = kp1[i].pt[1]
        kp[i,1] = kp1[i].pt[0]
        kp[i,2] = dep[int(kp[i][0]),int(kp[i][1])]
    return kp, des1

def match_kps(des1, des2):

    return matches


def combine_image_feature(kp1, des1, kp2, des2, matches):
    return kp, des

def calcule_T(matches, kp1, kp2):
    return T

def edge_detection(img):
    im = img.copy()
    model = 'opencv_extra-master/testdata/cv/ximgproc/model.yml'

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(30)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    for b in boxes:
        x, y, w, h = b
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    return edges

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def calu_dep(imagefile,depfile):
    my_file = Path(depfile)
    if not my_file.is_file():
        os.system("python predict.py "+ ckptfile + " " +imagefile+" "+depfile )
    return 0

def calu_dep_real_time(imagefile,depfile):
    print("start real time depth prediction.")
    os.system("python predictv2.py "+ ckptfile + " " +imagefile+" "+depfile )
    return 0


def average_matrix(Ms):
    # calcule the angle of translation by cv2.Rodrigues
    # the average these angles
    # but the problem is the Ms are not really transfomation trans_matrix
    # they have also distoration
    dd = np.zeros(3, dtype=np.float32)
    tt = np.zeros(3, dtype=np.float32)
    l = len(Ms)
    for i in range(l):
        M = Ms[i]
        t = M[0:3,3]
        dst, jacobian = cv2.Rodrigues(M[0:3,0:3])
        dd = dd + dst.T
        tt = tt + t
    dd = dd/l
    tt = tt/l
    return trans_matrix(dd,tt)

def average_matrix_v2(Ms):
    # calcule the angle of translation by cv2.Rodrigues
    # the average these angles (deleting the matrix with the largest variance)
    dd = []
    tt = []
    sum_dd = np.zeros(3, dtype=np.float32)
    sum_tt = np.zeros(3, dtype=np.float32)
    l = len(Ms)

    if l == 1 or l == 2:
        return Ms[0]

    for i in range(l):
        M = Ms[i]
        dst, jacobian = cv2.Rodrigues(M[0:3,0:3])
        tt.append(M[0:3,3])
        dd.append(dst.T)
        sum_dd = sum_dd + dst.T
        sum_tt = sum_tt + M[0:3,3]

    bad_idx = 10
    average_dd = sum_dd/l
    average_tt = sum_tt/l

    error_max = 0
    for i in range(l):
        error = np.sum(np.abs(dd[i]-average_dd) + np.abs(tt[i]-average_tt))
        if error >= error_max:
            error_max = error
            bad_idx = i

    dd_best = np.zeros(3, dtype=np.float32)
    tt_best = np.zeros(3, dtype=np.float32)
    n = 0
    for i in range(l):
        if i != bad_idx:
            dd_best = dd_best + dd[i]
            tt_best = tt_best + tt[i]
            n = n + 1
    dd_best = dd_best/n
    tt_best = tt_best/n
    return trans_matrix(dd_best,tt_best)

def average_matrix_v3(Ms):
    # calcule the average of matrix directly
    # with deleting the matrix with largest error
    average = np.zeros((3,4), dtype=np.float32)
    l = len(Ms)
    for i in range(l):
        average = average + Ms[i]
    average = average/l

    if l == 2 or l == 1:
        final_M = average
    else:
        error_max = 0
        for i in range(l):
            error = np.sum(np.abs(Ms[i] - average))
            if error > error_max:
                error_max = error
                idx_max = i
        final = np.zeros((3,4), dtype=np.float32)
        for i in range(l):
            if i != idx_max:
                final = final + Ms[i]
        final_M = final/(l-1)

    return final_M

def adjust_m(M):
    Mm = M[0:3,0:3]
    u, s, vh = np.linalg.svd(Mm)
    ss = np.eye(3,3)
    n_M = np.dot(u,np.dot(ss,vh))
    M[0:3,0:3] = n_M
    return M

def mut_trans(M1, M2):
    M1 = np.concatenate((M1, np.array([[0,0,0,1]])), axis=0)
    M2 = np.concatenate((M2, np.array([[0,0,0,1]])), axis=0)
    M = np.dot(M1,M2)
    return M[0:3,:]

def refine_seg_3D(res,size=60):
    m=np.size(res,0)
    perc=np.floor((m*size)/100)
    depth=res[:,2]
    nmean=list()
    difference=list()
    for i in range(80):
        bootstrap=np.random.randint(0, m, int(perc))
        echantillon=depth[bootstrap]
        nmean.append(echantillon.mean())
        difference.append(echantillon.max()-echantillon.min())
    nmean=np.array(nmean)
    difference=np.array(difference)
    mdiff=difference.mean()
    mmean=nmean.mean()
    intervalle=mmean+mdiff/2
    ress=np.zeros([m,6])
    t = 0
    for i in range(m):
        if res[i,2]<intervalle:
            ress[t,:] = res[i,:]
            t = t + 1
    return ress[0:t,:]

def add_image(imagefile, depfile,ddd=True, seg_img=False, seg_dep=False):
    print('Start matching the image in ', imagefile)
    image0 = cv2.imread(imagefile)
    image = cv2.resize(image0, (newWidth, newHeight))
    dep = np.load(depfile)

    images = glob.glob('keys/*.jpg')
    Ms = []
    l = len(images)
    for i in range(max(1,l-2), l+1):
        img0 = cv2.imread( 'keys/'+str(i)+'.jpg')
        img0 = cv2.resize(img0, (newWidth, newHeight))
        pts0, pts, kp1, kp2, matches, matchesMask = sift_points(img0,image,0.75)
        d0 = np.load('keys/deps/0'+str(i)+'.npy')
        #dep, _ = corres_imgs(image,img0,dep,d0)
        print('   matching the image with image '+ str(i))
        print('      -> the translation matrix is :')
        if len(pts) != 0:
            pts0 = verttt(pts0,d0[0,:,:,0] ,1)
            pts = verttt(pts,dep[0,:,:,0] ,1)
            retval, M, inliers = cv2.estimateAffine3D(pts, pts0, ransacThreshold=12)
            #M = rigid_transform_3D(pts.reshape(-1,3), pts0.reshape(-1,3))
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
    #M = average_matrix(Ms)
    if seg_img:
        image = mask_img_rgb(image, l+1)

    if seg_dep:
        dep = np.load(depfile[0:10]+depfile[11:])
    if len(Ms) != 0:
        M = Ms[len(Ms)-1]
        if ddd:
            res = vert(image, del_all(image,dep[0,:,:,0]), M, False, False)
        else:
            res = vert(image, dep[0,:,:,0], M,  False, False)
        if seg_dep:
            res = refine_seg_3D(res)
        write_ply('keys/res/'+str(l+1) +'.ply', res)
        np.save('keys/Ms/'+str(l+1) +'.npy' , M)
        cv2.imwrite('keys/'+str(l+1) +'.jpg',image0)
        print(' the process of image '+ str(l+1) + ' is finished.')
    else:
        print(' cannot match image '+ str(l+1) + '.')
    print(' ')
    return 0

def dep_seg(depfile, maskfile, savedep):
    dep = np.load(depfile)
    mask = cv2.imread(maskfile)
    newHeight = 128
    newWidth = 160
    mask = cv2.resize(mask, (newWidth, newHeight))
    for i in range(newHeight):
        for j in range(newWidth):
            if mask[i][j][2] == 0 :
                dep[0,i,j,0] = 0
    np.save(savedep , dep)
    return 0

def mask_img_rgb(image, i):
    maskfile =  'keys/seg/1'+str(i)+'_mask.png'
    mask = cv2.imread(maskfile)
    mask = cv2.resize(mask, (newWidth, newHeight))
    for i in range(newHeight):
        for j in range(newWidth):
            if mask[i][j][2] != 0 :
                image[i,j,0] = 0
                image[i,j,1] = 0
                image[i,j,2] = 255
    return image

def delete_module(modname, paranoid=None):
    from sys import modules
    try:
        thismod = modules[modname]
    except KeyError:
        raise ValueError(modname)
    these_symbols = dir(thismod)
    if paranoid:
        try:
            paranoid[:]  # sequence support
        except:
            raise ValueError('must supply a finite list for paranoid')
        else:
            these_symbols = paranoid[:]
    del modules[modname]
    for mod in modules.values():
        try:
            delattr(mod, modname)
        except AttributeError:
            pass
        if paranoid:
            for symbol in these_symbols:
                if symbol[:2] == '__':  # ignore special symbols
                    continue
                try:
                    delattr(mod, symbol)
                except AttributeError:
                    pass
