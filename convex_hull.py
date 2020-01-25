import numpy as np 
import cv2

def gift_wrap(points):
    
    leftmost = (points[np.argsort(points[:,0])])[0]
    currentvertex = leftmost.copy()
    nextvertex = None
    nextpoint = None
    vector1 = None
    vector2 = None
    edges = []
    index = np.where(points == leftmost)
    ix = index[0][0]

    flag = 1

    while not(np.allclose(currentvertex , leftmost)) or flag == 1:
        flag = 0
        nextvertex = points[ix - 1]
        for i in range(len(points)):
            nextpoint = points[i]
            vector1 = nextpoint - currentvertex
            vector2 = nextvertex - currentvertex
            if np.cross(vector2,vector1) > 0:
                nextvertex = nextpoint.copy()      
        edges.append([currentvertex , nextvertex])
        currentvertex = nextvertex.copy()
        index = np.where(points == currentvertex)
        ix = index[0][0]

    return edges

if __name__ == "__main__":

    import line

    size =750
    img = np.zeros((size,size),np.uint8)
    rgb = np.zeros((size,size,3),np.uint8)
    yl = size - 1
    no_of_pts= 30
    points = []

    for i in range(no_of_pts):
        points.append(np.random.random((2)) * (size - 1))

    points = np.array(points,np.int32)


    for x,y in points:
        rgb[yl-y,x,1:]=255


    edges = gift_wrap(points)

    for edge in edges:
        img = line.draw(img = img , x1=edge[0][0] , x2=edge[1][0] , y1=edge[0][1] , y2=edge[1][1])

    rgb[:,:,0] = img

    cv2.imshow("convex_hull",rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()