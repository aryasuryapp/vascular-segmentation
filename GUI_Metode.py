import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from math import log10, sqrt
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches

st.title('Image Segmentation')

st.header('Input Image')
file = './raw_images/frame1.jpg'
raw_image = cv2.imread(file)
# input_image = raw_image.copy()
# raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
st.image(raw_image)

st.header('Cropped Image')
cropped_image = raw_image[109:(raw_image.shape[0]-148), 204:(raw_image.shape[1]-204)]
st.image(cropped_image)

st.header('Circular Hough Transform')
st.subheader('Convert to gray')
gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
st.image(gray)

st.subheader('Blur Image using Median Blur')
col1, col2, col3= st.columns(3)

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

with col1:
    st.write('kernel 3x3')
    blurred1 = cv2.medianBlur(gray, 3)
    st.image(blurred1)
    value = PSNR(gray, blurred1)
    value = round(value, 3)
    st.write(f"PSNR value is {value} dB")

with col2:
    st.write('kernel 5x5')
    blurred2 = cv2.medianBlur(gray, 5)
    st.image(blurred2)
    value = PSNR(gray, blurred2)
    value = round(value, 3)
    st.write(f"PSNR value is {value} dB")

with col3:
    st.write('kernel 7x7')
    blurred3 = cv2.medianBlur(gray, 7)
    st.image(blurred3)
    value = PSNR(gray, blurred3)
    value = round(value, 3)
    st.write(f"PSNR value is {value} dB")

st.subheader('Map Circle')
rows = blurred1.shape[0]
circles = cv2.HoughCircles(blurred1, cv2.HOUGH_GRADIENT, 1, rows / 4,
                            param1=100, param2=30,
                            minRadius=int(25), maxRadius=int(70))
detected_circles = np.uint16(np.around(circles))
radius = []
count = 1
mapped = cropped_image.copy()
for (x, y ,r) in detected_circles[0, :]:
    radius.append(r)
    cv2.circle(mapped, (x, y), r, (0, 0, 255), 2) 
    cv2.circle(mapped, (x, y), 2, (0, 255, 255), 3) # titik tengah
    cv2.putText(mapped, str(count), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 255), 2, cv2.LINE_AA)
    count = count + 1
st.image(mapped)

st.subheader('Select Circle by Darkest Pixels')
st.write('Low value mean more dark')
res = []
for z in range(detected_circles.shape[1]):
  side = round(detected_circles[0][z][2]*np.sqrt(2))

  start_x = detected_circles[0][z][0]-round(side/2)
  start_y = detected_circles[0][z][1]-round(side/2)

  sum = 0
  for x in range(start_x,start_x+side,1):
    for y in range(start_y,start_y+side,1):
      sum = sum+gray[y][x]

  res.append(sum/(side*side)) 
np.round(res, 3)
res_sorted = res.copy()
res_sorted.sort()
index_circle = res.index(res_sorted[0])
x = np.arange(1,len(res)+1,1)
df = pd.DataFrame(list(zip(x, res)), columns =['Index Circle', 'Pixels Value'])
st.dataframe(df)
# st.dataframe(df.style.highlight_min(axis=1))
darkest = cropped_image.copy()
x1, y1 , r1 = detected_circles[0, index_circle]
cv2.circle(darkest, (x1, y1), r1, (0, 0, 255), 2) 
st.image(darkest)

st.header('Image Localization')
cleared = cropped_image.copy()
for x in range(cleared.shape[1]):
  for y in range(cleared.shape[0]):
    if list(cleared[y][x]) ==  [255,0,0] :
      cleared[y][x] = (255,255,255)

localized = cleared.copy()
cv2.circle(cleared, (x1, y1), r1, (255, 0, 0), 2)
for x in range(cleared.shape[1]):
  for y in range(cleared.shape[0]):
    if list(cleared[y][x]) ==  [255,0,0] :
      break
    localized[y][x] = (255,255,255)
for x in range(cleared.shape[1]):
  for y in reversed(range(cleared.shape[0])):
    if list(cleared[y][x]) ==  [255,0,0] :
      break
    localized[y][x] = (255,255,255)
col1, col2= st.columns(2)
with col1:
    st.image(localized)
with col2:
    fig, ax = plt.subplots()
    ax.hist(localized.ravel(),256,[0,250])
    ax.set_xlabel('pixel value')
    ax.set_ylabel('number of pixels')
    st.pyplot(fig)


st.header('Contrast Streching')
gray2 = cv2.cvtColor(localized, cv2.COLOR_BGR2GRAY)
alpha_input = st.slider('Pick Gain',1.0,10.0,value=3.0, step=0.1)
beta_input = st.slider('Pick Bias',0,100,value=80)
contrast_fix = cv2.convertScaleAbs(gray2, alpha=alpha_input, beta=beta_input)
col1, col2= st.columns(2)
with col1:
    st.image(contrast_fix)
with col2:
    fig, ax = plt.subplots()
    ax.hist(contrast_fix.ravel(),256,[0,250])
    ax.set_xlabel('pixel value')
    ax.set_ylabel('number of pixels')
    st.pyplot(fig)

st.header('Otsu Thresholding')
st.subheader('Apply Gaussian Blur')
col1, col2, col3= st.columns(3)
with col1:
    st.write('kernel 3x3')
    blur_otsu1 = cv2.GaussianBlur(contrast_fix,(3,3),0)
    st.image(blur_otsu1)
    value = PSNR(gray2, blur_otsu1)
    value = round(value, 3)
    st.write(f"PSNR value is {value} dB")

    fig, ax = plt.subplots()
    ax.hist(blur_otsu1.ravel(),256,[0,250])
    ax.set_xlabel('pixel value')
    ax.set_ylabel('number of pixels')
    st.pyplot(fig)

with col2:
    st.write('kernel 5x5')
    blur_otsu2 = cv2.GaussianBlur(contrast_fix,(5,5),0)
    st.image(blur_otsu2)
    value = PSNR(gray2, blur_otsu2)
    value = round(value, 3)
    st.write(f"PSNR value is {value} dB")

    fig, ax = plt.subplots()
    ax.hist(blur_otsu2.ravel(),256,[0,250])
    ax.set_xlabel('pixel value')
    ax.set_ylabel('number of pixels')
    st.pyplot(fig)

with col3:
    st.write('kernel 7x7')
    blur_otsu3 = cv2.GaussianBlur(contrast_fix,(15,15),0)
    st.image(blur_otsu3)
    value = PSNR(gray2, blur_otsu3)
    value = round(value, 3)
    st.write(f"PSNR value is {value} dB")

    fig, ax = plt.subplots()
    ax.hist(blur_otsu3.ravel(),256,[0,250])
    ax.set_xlabel('pixel value')
    ax.set_ylabel('number of pixels')
    st.pyplot(fig)

st.subheader('Apply Binary Tresholding')
otsu_value,otsu_image = cv2.threshold(contrast_fix,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
st.write(f"Otsu value: {otsu_value}")
col1, col2 = st.columns(2)
with col1:
    st.image(otsu_image)
with col2:
    fig, ax = plt.subplots()
    ax.hist(otsu_image.ravel(),256,[0,255])
    ax.set_xlabel('pixel value')
    ax.set_ylabel('number of pixels')
    st.pyplot(fig)

st.header('Region Properties')
st.subheader('Image Labeling')
label_image = label(otsu_image)
fig, ax = plt.subplots()
ax.imshow(label_image)
st.pyplot(fig)

st.subheader('Draw Bounding Box')
segmented_image = cropped_image.copy()
area = 0
for region in regionprops(label_image):
    if region.area > area:
            area = region.area
            taken_region = region
minr, minc, maxr, maxc = taken_region.bbox
rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
fig, ax = plt.subplots()
ax.imshow(label_image)
ax.add_patch(rect)
st.pyplot(fig)

cv2.rectangle(segmented_image, (minc, minr), (maxc, maxr), (0, 0, 255), 2)
cv2.putText(segmented_image, 'height=' + str(maxr-minr)+ 'px', (minc, maxr + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(segmented_image, 'width=' + str(maxc-minc)+ 'px', (minc, maxr + 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
st.image(segmented_image)