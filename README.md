# Deep Learning Service Project (Fall 2023)
![image](https://github.com/datascienceclubUVU/service-project/assets/111081544/01f02967-b9cf-4054-88f6-8dfabf0f3800)

### How the Model Works
1. Start with an image of a character of text:
- ![output](https://github.com/datascienceclubUVU/service-project/assets/111081544/f8f3664c-d88b-4364-b0f5-3427869bf5e2)
2. Convert the image between RGB/BGR and grayscale using the _**cvtColor**_ function from the _**cv2**_ library:
- ![image](https://github.com/datascienceclubUVU/service-project/assets/111081544/a94401d4-7557-4278-a6db-299ad1b5e7c0)
3. Use an Adaptive Thresholding approach where the threshold value = Gaussian weighted sum of the neighborhood values - constant value. In other words, it is a weighted sum of the blockSize^2 neighborhood of a point minus the constant. in this example, we are setting the maximum threshold value as 255 with the block size of 155 and the constant is 2.
- ![image](https://github.com/datascienceclubUVU/service-project/assets/111081544/aedb7c73-46c5-4497-8ead-84850c0101fc)
4. Create a 3x3 matrix of ones to generate an image kernel. An _**image kernel**_ is a small matrix used to apply effects like the ones you might find in Photoshop or Gimp, such as blurring, sharpening, outlining or embossing. They're also used in machine learning for 'feature extraction', a technique for determining the most important portions of an image.
5. The basic idea of erosion is just like soil erosion only, it erodes away the boundaries of foreground object (Always try to keep foreground in white). It is normally performed on binary images. It needs two inputs, one is our original image, second one is called structuring element or kernel which decides the nature of operation. A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).
- ![image](https://github.com/datascienceclubUVU/service-project/assets/111081544/1a031426-02e2-4910-b4cb-c3ad72c88a65)
6. The basic idea of dilation is accentuating the features of the images. Whereas erosion is used to reduce the amount of noise in the image, dilation is used to enhance the features of the image.
- ![image](https://github.com/datascienceclubUVU/service-project/assets/111081544/898f6254-fa96-4292-a847-b0dbc56f2535)
7. Traditionally, a line can be represented by the equation **_y=mx + b_** (where **_m_** is the slope and **_b_** is the intercept). However, a line can also be represented by the following equation: **_r= x(cos0) + y(sin0)_** (where **_r_** is the distance from the origin to the closest point on the straight line. **_(r,0)_** corresponds corresponds to the **_Hough space_** representation of the line. In this case, **_0_** is known as **_theta_**.
  
- For a given point in a two-dimensional space (think of a basic x- and y-axis graph), there can be an infinite number of straight lines drawn through the point. With a **_Hough Transform_**, you draw several lines through the point to create a table of values where you conclude "for given theta (angle between the x-axis and r-line that will match with the closest point on the straight line), we can expect this "r" value".
- Once you have created your table of values for each point on a given two-dimensional space, you compare the r-values on each theta for each given point and select the r and theta where the difference between the point is the least (this means the line best represents the points on the space).
