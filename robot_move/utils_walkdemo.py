import numpy as np
import sys
import cv2
import glob
import os
import time
from utils_bb import *
import skimage
from skimage import segmentation
from skimage.future import graph
newHeight = 128
newWidth = 160
def draw_boundbox_v2(index_fil = index_file):
    t = 0
    res = np.zeros((1,6))
    index = np.load(index_fil)
    for idx in index:
        top = idx['top_coord']
        z_max = idx["tall"][0]
        z_min = idx["tall"][1]
        label = idx["label"]
        off_set = idx["offset"]
        if label.split("_")[0] == 'floor' or label.split("_")[0] == 'wall':
            continue
        res1 = draw_polygen(top,z_max,z_min,off_set,idx["label"])
        t = t + 1
        res = np.concatenate((res,res1), axis=0)
    write_ply('keys/res/bb_'+ str(t) +'.ply', res[1:,:])
    return 0

def extraction_floor(img_seg):
    img_seg=img_seg[:,:,0]
    resultat=np.zeros((np.size((img_seg),0),np.size((img_seg),1)))
    hist=plt.hist(img_seg, bins = np.unique(img_seg))
    x=np.zeros(np.size(hist[0],0))
    for i in range(np.size(hist[0],0)):
        array=np.argsort(hist[0][i])
        x[i]=array[len(array)-1]
        array=np.zeros(np.size(hist[0],1))
    y=stats.mode(x)
    result=hist[1][np.int(y[0])]
    print(result)
    resultat[(img_seg>result-5) & (img_seg<result+5)]=55
    return(resultat)


def reset_delete(if_all=False,ggg=False):
    images = glob.glob('keys/images/img/*.*p*')
    for file in images:
        os.remove(file)
        images = glob.glob('keys/Ms/*.*')
    for file in images:
        os.remove(file)

    if if_all:
        images = glob.glob('keys/images/*.jpg')
        for file in images:
            os.remove(file)
        images = glob.glob('keys/deps/dep1/*.npy')
        for file in images:
            os.remove(file)
    if ggg:
        images = glob.glob('keys/objs/*')
        for file in images:
            os.remove(file)


def get_map_2d(res):
    map2d = np.zeros((200,200), dtype = np.int8)
    for i in range(len(res)):
        if 0 < res[i, 0] < 0.5:
            y = res[i,1]*50 + 100
            z = res[i,2]*50 + 100
            y = min(max(int(y),0),199)
            z = min(max(int(z),0),199)
            map2d[y,z] = 1
    tt = int(time.clock()*100)
    np.save('keys/bbs/'+str(tt)+'.npy', map2d)
    return map2d


def calcul_l(map2d, alpha):
    alpha1 = alpha - np.pi/15
    alpha2 = alpha + np.pi/15
    l = 0
    CON = True
    while(CON):
        l = l + 1
        x1 = int(l*np.cos(alpha1))+100
        y1 = int(l*np.sin(alpha1))+100
        x2 = int(l*np.cos(alpha2))+100
        y2 = int(l*np.sin(alpha2))+100
        x3 = int(l*np.cos(alpha))+100
        y3 = int(l*np.sin(alpha))+100
        CON = (map2d[x1,y1]==0 and map2d[x2,y2]==0 and map2d[x3,y3]==0 )
    return l


def distance_ahead(res, angle=np.pi):
    map2d = get_map_2d(res)
    #alpha = np.random.rand(1)*np.pi
    alpha = angle/2
    l = calcul_l(map2d, alpha)

    #if (l < 0.3*50):
    #    l1 = calcul_l(map2d, alpha-np.pi/10)
    #    l2 = calcul_l(map2d, alpha+np.pi/10)

    return l/50

def distance_ahead_one_image(res):
    map2d = get_map_2d(res)
    for z in range(100):
        t = map2d[100,z+100] + map2d[103,z+100] + map2d[97,z+100]
        if (map2d[0,z] != 0):
            return z

def calcul_M_round1(image0, dep, mtx, dst, num=10):
    images = glob.glob('keys/images/img/*.jpg')
    l = len(images)
    if num == 1:
        angle = 0
    else:
        angle = np.pi*l*(1.0)/(num-1)
    r = np.array([angle,0,0])
    t = np.zeros(3)
    M = trans_matrix(r,t)
    return M, l

def calcul_M_small_range(image0, dep, mtx, dst, num=10):
    images = glob.glob('keys/images/img/*.jpg')
    l = len(images)
    if num == 1:
        angle = 0
    else:
        angle = (np.pi*900/1900)*l*(1.0)/(num-1) + (np.pi*500/1900)
    r = np.array([angle,0,0])
    t = np.zeros(3)
    M = trans_matrix(r,t)
    return M, l


def vert_alphabot_fcn(depth,mtx):
    num = 20
    dep = np.zeros(20)
    kt = np.linalg.inv(mtx)
    for i in range(num):
        temp = depth[127-i,:]
        temp = reject_outliers(temp)
        dep[i] = np.mean(temp)
    #dep[0] = depth.min() + 0.4
    constant = (kt[0,0]*127 + kt[0,2])*dep[0]

    t = np.zeros(19)
    for i in range(1,num):
        if (dep[i]-dep[0]) == 0:
            t[i-1] = 20
        else:
            t[i-1] = (constant/(kt[0,0]*(127-i) + kt[0,2]) - dep[0])/(dep[i]-dep[0])
    for i in range(4):
        t = reject_outliers(t)
    #variance = np.sum(np.square(t- np.mean(t)))
    #print('    --- variance is '+ str(variance))
    return np.mean(t), dep[0]

def vert_alphabot(img,D, T,K, inv = True):
    l = img.shape[0]
    h = img.shape[1]
    t = 0
    temp = np.zeros((l*h, 6))
    T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)
    if inv:
        T = np.linalg.inv(T)
    ktt,d0 = vert_alphabot_fcn(D, K)
    new_dep = D
    for i in range(l):
        for j in range(h):
            dep = ktt*(D[i,j] - d0) + d0
            #dep = dep * (1.5 - i*0.5/64)
            new_dep[i,j] = dep
            if dep < d0:
                dep = D[i,j]
            if 0.1<dep< 200000:
                #camera caliberation
                img_p = np.array([i, j, 1])
                real_p = np.dot(np.linalg.inv(K),(img_p)*dep)
                real_p = np.append(real_p, 1)
                aa = np.dot(T, real_p).reshape(1,4)
                temp[t,0:3] = aa[0,0:3]/aa[0,3]
                temp[t,3:6] = img[i,j]
                t = t+1
    #if d0 == 0:
        #new_dep = None
    return temp[0:t,:], new_dep, d0

def add_image_demo(imagefile, depfile, old_res,num,seg = False,small_range=True,iteration=0):
    #print('Start matching the image ', imagefile)
    image0 = cv2.imread(imagefile)
    dep = np.load(depfile)

    mtx = np.eye(3)
    mtx[0,0]=200
    mtx[1,1]=200
    mtx[0,2]=64
    mtx[1,2]=80
    dst = np.zeros((1,5))

    if small_range:
        M,l = calcul_M_small_range(image0, dep, mtx, dst, num)
    else:
        M,l = calcul_M_round1(image0, dep, mtx, dst, num)

    image = cv2.resize(image0, (newWidth, newHeight))
    if seg:
        make_bounding_box_ncut(M,l,dep,imagefile, mtx)

    # "image" is the new added image "dep" is its depth map
    if old_res is None:
        res_final, new_dep,d0 = vert_alphabot(image, dep.reshape((newHeight,newWidth)), M,mtx, False)
    else:
        #imgr, depr, res_out = projection_3d_2d_v1(old_res, M, mtx)
        #new_dep = combine_dep_known_rgb(image, dep.reshape((newHeight,newWidth)), imgr, depr)
        #res_new_image = vert_new(image, new_dep.reshape((newHeight,newWidth)), M,mtx, False)
        res_new_image, new_dep ,d0= vert_alphabot(image, dep.reshape((newHeight,newWidth)), M,mtx, False)

        res_final = np.concatenate((old_res,res_new_image), axis=0)
        #np.save('keys/Ms/'+str(l+1) +'.npy' , M)
    dirr = 'keys/images/img'+str(iteration)
    if not os.path.exists(dirr):
        os.makedirs(dirr)
    cv2.imwrite(dirr+'/'+str(l+1) +'.jpg',image0)
    np.save(dirr+'/'+str(l+1) +'.npy',new_dep)
    #print(' the process of image is finished.')
    if new_dep is None:
        return None,d0
    return res_final,d0


##### NCUT segmentation

def ncut_image(image):
    image = cv2.resize(image, (160, 128))
    labels1 = segmentation.slic(image, compactness=30, n_segments=400)
    g = graph.rag_mean_color(image, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g,thresh=0.05, num_cuts=10)
    out2 = skimage.color.label2rgb(labels2, image, kind='avg')
    lenn = len(glob.glob('keys/objs/*.jpg'))
    cv2.imwrite('keys/objs/'+str(lenn+1)+'.jpg', out2)
    return labels2

def det_wall(labels):
    idxs = np.unique(labels[0:50,:])
    idx_wall = []
    for i in idxs:
        mask = np.ma.masked_equal(labels, i).mask * 1
        if np.sum(mask) > 50*160*0.8:
            idx_wall.append(i)
    return idx_wall

def proportion_floor(seg_img):
    img=seg_img[98:128,:]
    GG=np.unique(img)
    prop=list()
    idx = list()
    for i in GG:
        x=img[img==i]
        prop.append(len(x))
        idx.append(i)
    prop=np.array(prop)
    maxi=np.argsort(prop)
    res = []
    res.append(idx[int(maxi[len(maxi)-1])])
    if len(maxi) > 2:
        res.append(idx[int(maxi[len(maxi)-2])])
    return res

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def calcule_color(mask, image):
    r = np.zeros(3, dtype=np.int)
    for i in range(3):
        temp = image[:,:,i]*mask
        temp0 = reject_outliers(temp[temp!=0])
        r[i] = int(np.mean(temp0))
    return str(r[0])+'_'+str(r[1])+'_'+str(r[2])

def intersection_line(tall1, tall2):
    a1,b1 = min(tall1), max(tall1)
    a2,b2 = min(tall2), max(tall2)
    min_len = min(b1-a1, b2-a2)
    if a2<=a1<b2 :
        if b1<=b2:
            return 1
        else:
            return (b2-a1)/min_len
    elif a2<b1<=b2:
        return (b1-a2)/min_len
    else:
        return 0

def update_bb_v3(index,bb, T, label):
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
        if ppinter.is_empty:
            continue
        area_all = p_old.union(p_new).area
        line_inter = intersection_line(bb_new["tall"], index[i]["tall"])
        if line_inter < 0.6:
            continue
        if ppinter.area > p_old.area*0.8 and line_inter > 0.8:
            print('   --one bb inside another v1')
            index[i] = bb_new
            return index
        if ppinter.area > p_new.area*0.8 and line_inter > 0.8:
            print('   --one bb inside another v2')
            return index
        if ppinter.area > area_all*0.6 and line_inter > 0.8:
            print('   --intersection of bb')
            index[i]["top_coord"] = list(ppinter.exterior.coords)
            index[i]["tall"] = (bb_new["tall"] + index[i]["tall"])/2
            index[i]["offset"] = choose_offset(bb_new["offset"], index[i]["offset"])
            return index
    print('   -- new bb')
    index.append(bb_new)
    return index

def vert_boundingbox(img, depth_masked, depth, K):
    l = img.shape[0]
    h = img.shape[1]
    t = 0
    temp = np.zeros((l*h, 6))
    ktt,d0 = vert_alphabot_fcn(depth, K)
    for j in range(h):
        tempp = depth_masked[:,j]
        mean_dep = np.mean(reject_outliers(tempp[tempp!=0]))
        for i in range(l):
            dep = depth_masked[i,j]
            if 0.1<dep and abs(dep-mean_dep)<0.2:
                if dep > d0:
                    dep = ktt*(dep - d0) + d0
                #camera caliberation
                img_p = np.array([i, j, 1])
                real_p = np.dot(np.linalg.inv(K),(img_p)*dep)
                temp[t,0:3] = real_p
                temp[t,3:6] = img[i,j]
                t = t+1
    return temp[0:t,:]

def find_bounding_box_alphabot(res):
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

def get_object_bb_alphabot(imagefile, dep_masked,depth,  mtx):
    image = cv2.imread(imagefile)
    image = cv2.resize(image, (newWidth, newHeight))
    res = vert_boundingbox(image, dep_masked.reshape((newHeight,newWidth)), depth, mtx)
    if len(res) == 0:
        return None,None
    else:
        bbres, bb = find_bounding_box_alphabot(res)
        return bbres,bb

def make_bounding_box_ncut(M,l,dep,imagefile, mtx):
    image = cv2.imread(imagefile)
    image = cv2.resize(image, (newWidth, newHeight))
    labels = ncut_image(image)
    idx_floor = proportion_floor(labels)
    idx_wall = det_wall(labels)
    print('    ----- index floor is '+ str(idx_floor))
    print('    ----- index wall is '+ str(idx_wall))
    idxs = np.unique(labels)
    index_file = 'keys/objs/index.npy'
    if os.path.isfile(index_file):
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
    else:
        index = None

    for i in idxs:
        mask1 = np.ma.masked_equal(labels, i)
        mask = mask1.mask
        label = calcule_color(mask, image)
        if (i in idx_floor):
            print('     ----- seen floor')
            label = 'floor_' + label
        elif idx_wall != [] and (i in idx_wall):
            print('     ----- seen wall')
            label = 'wall_' + label

        dep_masked = dep.reshape((newHeight ,newWidth))*(mask*1)
        dep_non_obj = dep.reshape((newHeight ,newWidth)) - dep_masked
        _, bb = get_object_bb_alphabot(imagefile, dep_masked,dep.reshape((newHeight ,newWidth)), mtx)

        index = update_bb_v3(index, bb, M, label)

        #save the point cloud of the masked object
        #res_obj = vert(image, dep_masked, M, False)
        #write_ply('keys/objs/'+ str(l+1) + label+'.ply', res_obj)

        #delete the points in the bounding box
        #res0 = delete_pts_in_bb(res0, bb)

        #np.save('keys/objs/'+ label +'.npy' , bb)
    index_filett = 'keys/objs/index'+str(l) +'.npy'
    with open(index_filett, 'wb') as f:
        pickle.dump(index, f)
    with open(index_file, 'wb') as f:
        pickle.dump(index, f)
    print('fin one image')
    return 0
