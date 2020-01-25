import numpy as np 
import math

def draw(ig,x1,x2,y1,y2,color = 255):
	
	img = ig.copy()
	xl=img.shape[1]-1
	yl=img.shape[0]-1

	img[yl - y1 , x1] = color
	img[yl - y2 , x2] = color
	
	quadrant = 1

	if x2 < x1 and y2 >= y1:
		quadrant = 2
		x2 = x1 + x1 - x2

	elif x2 < x1 and y2 < y1:
		quadrant = 3
		x2 = x1 + x1 - x2
		y2 = y1 + y1 - y2

	elif x2 >= x1 and y2 < y1:
		quadrant = 4
		y2 = y1 + y1 - y2

	if (x2 - x1) != 0:
		m=(y2-y1)/(x2-x1)

		t=math.atan(m)*180/np.pi

		#print("angle=",t)

	else:
		t = 90

	if t>45:

		h=x1
		x1=y1
		y1=h
		k=x2
		x2=y2
		y2=k


	dx=x2-x1
	dy=y2-y1

	x=x1
	y=y1

	#img[yl-y,x]=255

	d=dy-(dx/2)

	for x in range(x1+1,x2):

		if d<0:
			d=d+dy
		else:
			d=d+dy-dx
			y=y+1


		if quadrant == 2:

			if t>45:
				img[yl - x , y - (2 * (y - y1)) ] = color
			else:
				
				img[yl - y  , x - (2 * (x - x1))] = color

		elif quadrant == 4:

			if t>45:
				img[yl - x + 2*(x - x1), y ] = color
			else:
				img[yl - y + 2*(y - y1),x] = color


		elif quadrant == 3:	

			if t>45:
				img[yl - x  + 2*(x - x1), y - (2 * (y - y1)) ] = color
			else:
				img[yl - y + 2*(y - y1),x - (2 * (x - x1))] = color

		else:

			if t>45:
				img[yl-x,y] = color
			else:
				img[yl-y,x] = color


	return img

