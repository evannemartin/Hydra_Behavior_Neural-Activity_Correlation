import numpy as np
import tps
import time
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# Compute the angle

def get_angle(a,b,c):

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

# Open the data
points = pd.read_csv('../data/hydra_long_moit1.csv', header=None)
points.head()

# Extract just the data
row_data = points.loc[points[:][0] == '0'].index[0]
datas = points[row_data:].copy()

# Extract t
t = pd.to_numeric(datas[0]).to_numpy()

# Extract xy from different regions

xybodyleft = np.zeros((datas.shape[0], 2))
xybodyleft[:,0] = pd.to_numeric(datas[1]).to_numpy()
xybodyleft[:,1] = pd.to_numeric(datas[2]).to_numpy()

xybodyright = np.zeros((datas.shape[0],2))
xybodyright[:,0] = pd.to_numeric(datas[4]).to_numpy()
xybodyright[:,1] = pd.to_numeric(datas[5]).to_numpy()

xyhypo = np.zeros((datas.shape[0],2))
xyhypo[:,0] = pd.to_numeric(datas[7]).to_numpy()
xyhypo[:,1] = pd.to_numeric(datas[8]).to_numpy()

xyped = np.zeros((datas.shape[0],2))
xyped[:,0] = pd.to_numeric(datas[10]).to_numpy()
xyped[:,1] = pd.to_numeric(datas[11]).to_numpy()

xycenter = np.zeros((datas.shape[0],2))
xycenter[:,0] = (xybodyleft[:,0]+xybodyright[:,0])/2
xycenter[:,1] = (xybodyleft[:,1]+xybodyright[:,1])/2

# Compute angle
l = xyhypo.shape[0]
angle = np.zeros(l)
for i in range(l):
    angle[i] = get_angle(xyhypo[i],xycenter[i],xyped[i])
print(angle.shape)

# Smoothing

# Learn the spline interpolator. High alpha => more regularisation
thin = tps.ThinPlateSpline(alpha=50).fit(t[:, None], xybodyleft)
# Compute a smooth version of xy
xybodyleftsmooth = thin.transform(t[:, None])

thin = tps.ThinPlateSpline(alpha=50).fit(t[:, None], xybodyright)
xybodyrightsmooth = thin.transform(t[:, None])

thin = tps.ThinPlateSpline(alpha=50).fit(t[:, None], xyhypo)
xyhyposmooth = thin.transform(t[:, None])

thin = tps.ThinPlateSpline(alpha=50).fit(t[:, None], xyped)
xypedsmooth = thin.transform(t[:, None])

thin = tps.ThinPlateSpline(alpha=50).fit(t[:, None], xycenter)
xycentersmooth = thin.transform(t[:, None])

#Length and width
length = np.linalg.norm(xyhyposmooth-xypedsmooth, axis=1)
width = np.linalg.norm(xybodyrightsmooth-xybodyleftsmooth, axis=1)


print("Smoothing done")

# Segmenting hydra

d1 = np.zeros((xycenter.shape[0],2))
d2 = np.zeros((xycenter.shape[0],2))
d3 = np.zeros((xycenter.shape[0],2))
d4 = np.zeros((xycenter.shape[0],2))
d5 = np.zeros((xycenter.shape[0],2))
d6 = np.zeros((xycenter.shape[0],2))

d1p = np.zeros((xycenter.shape[0],2))
d2p = np.zeros((xycenter.shape[0],2))
d3p = np.zeros((xycenter.shape[0],2))
d4p = np.zeros((xycenter.shape[0],2))
d5p = np.zeros((xycenter.shape[0],2))
d6p = np.zeros((xycenter.shape[0],2))

p1p2 = np.zeros(xycenter.shape[0])
p1p3 = np.zeros(xycenter.shape[0])

moon = np.zeros((xycenter.shape[0],2))
sun = np.zeros((xycenter.shape[0],2))
moonp = np.zeros((xycenter.shape[0],2))
sunp = np.zeros((xycenter.shape[0],2))

realwidth = np.zeros(xycenter.shape[0])
realwidthp = np.zeros(xycenter.shape[0])

for i in range (xybodyleft.shape[0]) : #parcourt frame by frame
    print(i, end="\r")

    xcenter = xycentersmooth[i][0]  # to see if match perfectly with the points don't use smooth
    ycenter = xycentersmooth[i][1]
    xhypo = xyhyposmooth[i][0]
    yhypo = xyhyposmooth[i][1]
    xped = xypedsmooth[i][0]
    yped = xypedsmooth[i][1]
    xbodyleft = xybodyleftsmooth[i][0]
    ybodyleft = xybodyleftsmooth[i][1]
    xbodyright = xybodyrightsmooth[i][0]
    ybodyright = xybodyrightsmooth[i][1]

    #implementing d1
    a1 = ((ycenter-yhypo)/(xcenter-xhypo))
    b1 = yhypo-a1*xhypo
    d1[i,0] = a1
    d1[i,1] = b1

    #implementing d4
    a4 = (xhypo-xcenter)/(ycenter-yhypo)
    b4 = yhypo-a4*xhypo
    d4[i,0] = a4
    d4[i,1] = b4

    #implementing d2
    xmoon = -((xcenter-xbodyleft)*(20)/(width[i]/2))+xbodyleft
    ymoon = -((ycenter-ybodyleft)*(20)/(width[i]/2))+ybodyleft

    moon[i,0] = xmoon
    moon[i,1] = ymoon

    a2 = a1
    b2 = (ymoon)-a2*(xmoon)
    d2[i,0] = a2
    d2[i,1] = b2

    #implementing d3
    xsun = ((xbodyright-xcenter)*(20)/(width[i]/2))+xbodyright
    ysun = ((ybodyright-ycenter)*(20)/(width[i]/2))+ybodyright

    sun[i,0] = xsun
    sun[i,1] = ysun

    a3 = a1
    b3 = (ysun)-a3*(xsun)
    d3[i,0] = a3
    d3[i,1] = b3

    #implementing d5
    p1p2[i] = np.linalg.norm(xyhypo[i]-xycenter[i])
    xstar = xcenter+(xhypo-xcenter)/2
    ystar = ycenter+(yhypo-ycenter)/2

    a5 = a4
    b5 = ystar-a4*xstar
    d5[i,0] = a5
    d5[i,1] = b5

    #implementing d6
    a6 = a4
    b6 = ycenter-a6*xcenter
    d6[i,0] = a6
    d6[i,1] = b6

    #implementing d1prime
    a1p = np.abs((yped-ycenter)/(xped-xcenter))
    b1p = yped-a1p*xped
    d1p[i,0] = a1p
    d1p[i,1] = b1p

    #implementing d4prime
    #a4p=(xcenter-xped)/(yped-ycenter)
    a4p =- 1/a1p
    b4p = yped-a4p*xped
    d4p[i,0] = a4p
    d4p[i,1] = b4p

    #implementing d2prime
    xmoonp = -((xcenter-xbodyleft)*(20)/(width[i]/2))+xbodyleft
    ymoonp = -((ycenter-ybodyleft)*(20)/(width[i]/2))+ybodyleft

    moonp[i,0] = xmoonp
    moonp[i,1] = ymoonp

    a2p = a1p
    b2p = (ymoonp)-a2p*(xmoonp)
    d2p[i,0] = a2p
    d2p[i,1] = b2p


    #implementing d3prime
    xsunp = ((xbodyright-xcenter)*(20)/(width[i]/2))+xbodyright
    ysunp = ((ybodyright-ycenter)*(20)/(width[i]/2))+ybodyright

    sunp[i,0] = xsunp
    sunp[i,1] = ysunp

    a3p = a1p
    b3p = (ysunp)-a3p*(xsunp)
    d3p[i,0] = a3p
    d3p[i,1] = b3p

    #implementing d5prime
    p1p3[i] = np.linalg.norm(xycenter[i]-xyped[i])
    xstarp = xped+(xcenter-xped)*3/5
    ystarp = yped+(ycenter-yped)*3/5

    a5p = a4p
    b5p = ystarp-a4p*xstarp
    d5p[i,0] = a5p
    d5p[i,1] = b5p


    #implementing d6prime
    a6p = a4p
    b6p = ycenter-a6p*xcenter
    d6p[i,0] = a6p
    d6p[i,1] = b6p

    #computing realwidth
    realwidth[i] = np.linalg.norm(moon[i]-sun[i])
    realwidthp[i] = np.linalg.norm(moonp[i]-sunp[i])

print("d made")

# Extract intensity

cap = cv2.VideoCapture('../Data/Annotated/hydra_long_moit1DLC.mp4')
l = 0
m = 0

totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
lengthpic = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
widthpic = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

upintensities = np.zeros(totalframes)
lowintensities = np.zeros(totalframes)
midintensities = np.zeros(totalframes)


x = np.arange(widthpic)
y = np.arange(lengthpic)
x, y = np.meshgrid(x, y)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        # Conditions

        above_d3 = d3[l,0]*x+d3[l,1] < y
        below_d3 = d3[l,0]*x+d3[l,1] > y
        below_d2 = d2[l,0]*x+d2[l,1] > y
        above_d4 = d4[l,0]*x+d4[l,1] < y
        below_d5 = d5[l,0]*x+d5[l,1] > y
        above_d5 = d5[l,0]*x+d5[l,1] < y
        below_d6 = d6[l,0]*x+d6[l,1] > y
        above_d6 = d6[l,0]*x+d6[l,1] < y
        above_d3p = d3p[l,0]*x+d3p[l,1] < y
        below_d2p = d2p[l,0]*x+d2p[l,1] > y
        below_d4p = d4p[l,0]*x+d4p[l,1] > y
        above_d5p = d5p[l,0]*x+d5p[l,1] < y
        below_d5p = d5p[l,0]*x+d5p[l,1] > y
        above_d6p = d6p[l,0]*x+d6p[l,1] < y
        below_d6p = d6p[l,0]*x+d6p[l,1] > y

        inter_d3_d2 = np.logical_and(above_d3, below_d2)
        inter_d5_d4 = np.logical_and(above_d4, below_d5)
        inter_d5_d6 = np.logical_and(above_d5, below_d6)
        inter_d3p_d2p = np.logical_and(above_d3p, below_d2p)
        inter_d5p_d4p = np.logical_and(below_d4p, above_d5p)
        inter_d5p_d6p = np.logical_and(below_d5p, above_d6p)

        inter_d6_d6p = np.logical_and(below_d6p, above_d6)
        inter_d3_d2_d3p_d2p = np.logical_and(inter_d3_d2, inter_d3p_d2p)


        # Compute directly the sum
        upsum = (frame[np.logical_and(inter_d3_d2, inter_d5_d4)][...,1]).sum()
        frame[..., 1][np.logical_and(inter_d3_d2, inter_d5_d4)] = 0 #to avoid counting 2 times with midsum for overlapping cases

        lowsum = (frame[np.logical_and(inter_d3p_d2p, inter_d5p_d4p)][...,1]).sum()
        frame[..., 1][np.logical_and(inter_d3p_d2p, inter_d5p_d4p)] = 0

        midsum = (frame[np.logical_or(np.logical_and(inter_d3_d2, inter_d5_d6), np.logical_and(inter_d3p_d2p, inter_d5p_d6p))][...,1]).sum()
        midsum = midsum + (frame[np.logical_and(inter_d6_d6p, inter_d3_d2)][...,1]).sum() #add the triangle

        # If a rectangle disappear = issue
        if lowsum == 0:
            break
        if midsum == 0:
            break
        if upsum == 0:
            break

        # Changing the color to visually check
        frame[..., 0][np.logical_and(inter_d3_d2, inter_d5_d4)] = 255 #up
        frame[..., 1][np.logical_and(inter_d3p_d2p, inter_d5p_d4p)] = 255 #low
        frame[..., 2][np.logical_or(np.logical_and(inter_d3_d2, inter_d5_d6), np.logical_and(inter_d3p_d2p, inter_d5p_d6p))] = 255 #mid
        frame[..., 2][np.logical_and(inter_d6_d6p, inter_d3_d2_d3p_d2p)] = 255 #add the triangle


        if 420 < l < 430:  # To go faster to the problem 423
            frame_rate = 30
        else:
            frame_rate = 30


        # Display the resulting frame
        xb = int(tuple(xycentersmooth[l])[0])
        yb = int(tuple(xycentersmooth[l])[1])
        cv2.circle(frame, (xb,yb), 8, (0,0,0))
        im = cv2.resize(frame,(600,600))
        cv2.imshow('Frame',im)
        cv2.setWindowTitle('Frame', f'Frame {l} - {frame_rate}')


        # Compute the area = nb of pixels
        uparea = realwidth[l]*p1p2[l]/2
        lowarea = realwidthp[l]*p1p3[l]*3/5
        midarea = realwidth[l]*p1p2[l]/2 + realwidthp[l]*p1p3[l]*2/5

        # Compute the average
        upintensities[l] = upsum/uparea
        lowintensities[l] = lowsum/lowarea
        midintensities[l] = midsum/midarea

        print(l)
        l += 1

        # Press Q on keyboard to  exit
        if cv2.waitKey(1000//frame_rate) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Save arrays
np.save('upintensitieslong.npy',upintensities)
np.save('lowintensitieslong.npy',lowintensities)
np.save('midintensitieslong.npy',midintensities)

# Closes all the frames
cv2.destroyAllWindows()
