#Enter, segmented image, and the associated depth
#Return,resultat, that is the floor, if it's the floor, Object, if not
def extraction_floor(img_seg,dep):
    img_seg=img_seg[:,:,0]
    resultat=np.zeros((np.size((img_seg),0),np.size((img_seg),1)))
    resultat1=np.zeros((np.size((img_seg),0),np.size((img_seg),1)))
    hist=plt.hist(img_seg, bins = np.unique(img_seg))
    x=np.zeros(np.size(hist[0],0))
    for i in range(np.size(hist[0],0)):
        array=np.argsort(hist[0][i])
        #print(array)
        x[i]=array[len(array)-1]
        array=np.zeros(np.size(hist[0],1))
    y=stats.mode(x)
    result=hist[1][np.int(y[0])]
    #print(result)
    resultat[(img_seg>result-5) & (img_seg<result+5)]=1
    img=resultat*dep
    plt.imshow(img)
    #print("on entre")
    if are_we_on_the_floor(img)==True:
        return(resultat)
    else:
        frequencies = Counter(x)
        j=0
        h=np.zeros(np.size(np.unique(x)))
        for i in np.unique(x):
            h[j]=frequencies[i]
            j=j+1
        k=np.argsort(h)
        idx=k[len(k)-2]
        second=np.unique(x)[idx]
        result1=hist[1][np.int(second)]
        #print(result)
        resultat1[(img_seg>result1-2) & (img_seg<result1+2)]=1
        img1=resultat1*dep
        plt.imshow(img1)
        if are_we_on_the_floor(img1)==True:
            return(resultat1)
        else:
            return(print("OBJECT"),resultat)

def higher_column(dep_seg):
    Heigth = np.size(dep_seg,0)
    Width = np.size(dep_seg,1)
    f=list()
    column=list()
    for j in range(Width):
        i=0
        #print("ici")
        while ((i<127) & (dep_seg[i,j]==0)):
            i=i+1
        mm=dep_seg[i:Heigth-1,j]
        #print(mm)
        if ((np.size(np.nonzero(mm))==np.size(mm)) & (np.size(mm)!=0)):
            h=np.size(mm)
            f.append(h)
            column.append(j)
            print(h)
        #f.append(np.size(img[img[:,j]!=0],0))
    f=np.array(f)
    #print(f)
    column=np.array(column)
    idx=np.argsort(f)
    if np.size(f) == 0:
        return(None)
    else:
        higher=column[idx[len(idx)-1]]
        return(higher)






def are_we_on_the_floor(dep_seg):
    higher=higher_column(dep_seg)
    print(higher)
    if higher == None:
        return(False)
    test=np.zeros(np.size(dep_seg,1))
    test=dep_seg[:,higher]
    #print(test)
    kk=np.zeros(np.size(test!=0))
    kk= test[test!=0]
    print(kk)
    m=kk.max()-kk.min()
    print(m)
    if m>1.2:
        return(True)
    else:
        return(False)



def calcule_distance_from_seg(image, depth):
    image = cv2.resize(image, (160,128))
    kernel = np.ones((6,6),np.uint8)
    image1 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    """
    image1 = np.zeros((128,160,3))
    image1[:,:,0] = np.abs(image[:,:,0] - image[:,:,1])
    image1[:,:,1] = np.abs(image[:,:,1] - image[:,:,2])
    image1[:,:,2] = np.abs(image[:,:,2] - image[:,:,0])
    """
    ss = cv2.ximgproc.segmentation.createGraphSegmentation(0.5,250,180)
    seg = ss.processImage(image1)

    seg_test = seg[80:120, :]
    temp = np.unique(seg_test)
    idx = []
    for i in range(len(temp)):
        mask0 = np.ma.masked_equal(seg_test , temp[i])
        if np.sum(mask0.mask*1)/ (160*40) > 0.2:
            idx.append(temp[i])
    res = np.zeros((128,160))
    for i in range(len(idx)):
        res = res + np.ma.masked_equal(seg , idx[i]).mask*1

    floor = (depth.reshape(128,160)*(res))[90:120,:]
    objs = (depth.reshape(128,160)*(1-res))[50:120,:]
    try:
        d_obj = np.min(objs[objs!=0])
        d_flo = np.mean(floor[floor!=0])
    except ValueError:
        d_obj = 0
        d_flo = 0

    if np.sum((1-res)[80:120,40:120]) < (80*40*0.3) and np.sum(res[0:40,:]) < 10*20:
        print('go ahead')
        distance = np.min(depth[0,50:100 ,50:110 ,0]) - d_flo + 0.8
    elif d_obj - d_flo < 0.2:
        print('obstacle')
        distance = 0
    else :
        print('obstacle far')
        distance = d_obj - d_flo

    return distance, res, seg
