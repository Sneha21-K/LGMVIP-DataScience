import cv2


# - # get the image location and the image file name

# In[2]:


img_location = 'C:/Users/admin/Desktop/LGMVIP/TASK 3/'


# In[3]:


filename = 'sneha1.jpg'


# - # read in the image

# In[4]:


img = cv2.imread(img_location+filename)


# - # convert the image to gray scale

# In[5]:


gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# - # invert the image

# In[6]:


inverted_gray_image = 255 - gray_image


# - # blur the image by gaussian blur

# In[7]:


blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)


# - # invert the blurred image

# In[8]:


inverted_blurred_image = 255 - blurred_image


# - # create the pencil sketch image

# In[9]:


pencil_sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale = 256.0)


# - # show the image

# In[10]:


cv2.imshow('Original Image', img)


# In[11]:


cv2.imshow('New Image', pencil_sketch_image)


# In[12]:


cv2.waitKey(0)


# In[ ]:

