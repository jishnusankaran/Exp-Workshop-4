# DIPT-WORKSHOP-4 ( Coin Detection using OpenCV in Python ) 
# Name : JISHNUPRIYAN S
# Reg No : 212223240061

## Aim :
To detect and count coins in an image using OpenCV image processing and Simple Blob Detection.


## Algorithm :

1. Read the coin image and convert it from BGR to RGB format.
2. Convert the RGB image into a grayscale image.
3. Split the image into Red, Green, and Blue color channels.
4. Apply binary inverse thresholding on the Green channel.
5. Perform dilation followed by erosion using an 8×8 kernel to enhance the coin regions.
6. Configure and apply OpenCV's SimpleBlobDetector using circularity, convexity, and inertia filters.
7. Detect the coin centers, draw circles around the detected coins, and count the detected blobs.


## Program : 

```py

import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('coins.png')
image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
 


imageCopy = image.copy()
plt.imshow(image);
plt.title("Original Image")
plt.show()
# Expected output


imageGray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
plt.figure(figsize=(12,12))
plt.subplot(121);plt.imshow(image);plt.title("Original Image")
plt.subplot(122); plt.imshow(imageGray,cmap='gray');plt.title("Grayscale Image"); plt.show()
# Expected output



# Split cell into channels
# Store them in variables imageB, imageG, imageR
imageR,imageG,imageB=cv2.split(image)
plt.figure(figsize=(20,12))
plt.subplot(141);plt.imshow(image);plt.title("Original Image")
plt.subplot(142);plt.imshow(imageB,cmap='gray');plt.title("Blue Channel")
plt.subplot(143);plt.imshow(imageG,cmap='gray');plt.title("Green Channel")
plt.subplot(144);plt.imshow(imageR,cmap='gray');plt.title("Red Channel");
plt.show()
# Expected output



ret, thresh_inv = cv2.threshold(imageG, 20,255, cv2.THRESH_BINARY_INV)
 
 
plt.imshow(thresh_inv,cmap='gray');
plt.title("Original Image")
plt.show()
kernel=np.ones((8,8),dtype=np.uint8)
dilution=cv2.dilate(thresh_inv,kernel,iterations=1)
plt.imshow(dilution,cmap='gray');plt.title('Dilated Image Iteration 2');plt.show()
# Expected output

erosion=cv2.erode(dilution,kernel,iterations=1)
 
plt.imshow(erosion,cmap='gray');plt.title("Eroded Image");plt.show()
# Expected output



# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.8



# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(erosion)
# Print number of coins detected

print(f"Number of coins detected: {len(keypoints)}")


for k in keypoints:
    x,y =k.pt
    x=int(round(x))
    y=int(round(y))

    cv2.circle(image,(x,y),5,(255,0,0),-1)

    diameter = k.size
    radius = int(round(diameter/2))

    cv2.circle(image,(x,y),radius,(0,255,0),2)
plt.imshow(image,cmap="gray")
plt.title("Fianl Image")
plt.show()


```


## Output : 


<img width="512" height="510" alt="image" src="https://github.com/user-attachments/assets/a1502b2d-ca17-4bfe-b069-fa53da7bc267" />


<img width="1138" height="585" alt="image" src="https://github.com/user-attachments/assets/84f602e9-0720-482b-87c9-1cee0826ad83" />


<img width="1245" height="346" alt="image" src="https://github.com/user-attachments/assets/028559c7-520e-47bc-bd9d-cd515d2d4881" />

<img width="441" height="840" alt="image" src="https://github.com/user-attachments/assets/cbd1ab34-a1b3-491c-93da-9fa0134c04e9" />


<img width="422" height="427" alt="image" src="https://github.com/user-attachments/assets/6ed78e4f-975c-479d-b6fd-6d530ce453c2" />



<img width="402" height="428" alt="image" src="https://github.com/user-attachments/assets/3efc29f8-5061-4d92-b557-7d3804f62a19" />


## Result : 
The coins were successfully detected and marked in the image using Simple Blob Detection.
