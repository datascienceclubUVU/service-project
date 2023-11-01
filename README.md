# Machine Learning Hackathon (Fall 2023)
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

