import numpy as np
import cv2
import matplotlib.pyplot as plt

image=cv2.imread('/home/rahul/Pictures/cornerDetectionImage.jpg',0)
edges=cv2.Canny(image,120,200)
def hough(img):

    #defining the range of rho and theta
    thetas=np.deg2rad(np.arange(-90.0 , 90.0))
    width,height=img.shape
    diag=np.ceil(np.sqrt(width*width + height*height))
    diag=int(diag)
    rhos=np.linspace(-diag,diag,diag*2)
    # defining trigo functions
    cos_t=np.cos(thetas)
    sin_t=np.sin(thetas)
    n=len(thetas)

    # constructing accumulator
    accumulator=np.zeros((2*diag,n),dtype=np.uint64)

    good_y,good_x=np.nonzero(img)

    for i in range(len(good_x)):
        x=good_x[i]
        y=good_y[i]

        for theta in range(n):
            rho=round(x * cos_t[theta]+y * sin_t[theta])+diag
            rho=int(rho)
            accumulator[rho,theta] += 1
    return accumulator,thetas,rhos
a , t, r =hough(edges)
#cv2.imshow('accumulator',a)
#cv2.imshow('thetas',t)
#cv2.imshow('rhos',r)


#showing hough lines
def show_hough(img,accumulator):


    plt.imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(t[-1]), np.rad2deg(t[0]), r[-1], r[0]])
    plt.savefig('/home/rahul/Pictures/output.png', bbox_inches='tight')
    plt.show()

show_hough(image,a)

op=cv2.imread('/home/rahul/Pictures/output.png',0)

cv2.imshow('edges',edges)
cv2.imshow('output',op)
cv2.waitKey(0)
cv2.destroyAllWindows()

c=max(np.ravel(a))
print c
d=a.shape[0]
theta=a.shape[1]
for q in range(d):
    for w in range(theta):
        if a[q,w]==c:
            print (q,w)
sin = np.sin(w)
cos = np.cos(w)
print(sin,cos)
x0 = q*cos
y0 = q*sin
x1 = int(x0 + 1000*(-sin))
y1 = int(y0 + 1000*(cos))
x2 = int(x0 - 1000*(-sin))
y2 = int(y0 - 1000*(cos))
print x1,y1,x2,y2
