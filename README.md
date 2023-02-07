# Lane detection for autonomous vehicles
Lane detection of autonomous vehicles

# Installation required -
    sudo apt-get install python3-pip
    pip3 install opencv-python
    pip3 install scipy

# How to run -
    python3 lane_detection.py --input_image <image_path>
Also one can find other command line arguments using -

    python3 lane_detection.py --help

# Steps involved -
**Read image -**<br/>

<!--![Result 1](/images/lane.jpg){:height="50%" width="50%"}-->
<img src="/images/lane.jpg" width="800" height="600">

**Canny edge detection -**<br/>

<!--![Result 2](/images/canny.jpg)-->
<img src="/images/canny.jpg" width="800" height="600">

**Image mask -**<br/>
If first mask don't work then i am using 2nd mask and so on. In worst case, i will use four mask.<br/>
<!--![Mask 1](/images/mask1.jpg) ![Mask 2](/images/mask2.jpg) ![Mask 3](/images/mask3.jpg) ![Mask 4](/images/mask4.jpg)-->
<img src="/images/mask1.jpg" width="200" height="150"> <img src="/images/mask2.jpg" width="200" height="150"> <img src="/images/mask3.jpg" width="200" height="150"> <img src="/images/mask4.jpg" width="200" height="150">

**Find ROI using image mask -**<br/>
<!--![Mask canny](/images/Masked_canny.jpg)-->
<img src="/images/Masked_canny.jpg" width="800" height="600">

**Detecting, classifying and averaging left and right lines with the help of hough transform -**<br/>
<!--![lines](/images/lines.jpg)-->
<img src="/images/lines.jpg" width="800" height="600">

**Cropping and giving final result -**<br/>
<!--![output](/images/output.jpg)-->
<img src="/images/output.jpg" width="800" height="600">

# Some other Results -

<img src="/images/output4.jpg" width="800" height="600">
<img src="/images/output2.jpg" width="800" height="600">
<img src="/images/output3.jpg" width="800" height="600">
<img src="/images/output1.jpg" width="800" height="600">


# Conclusion -
I implimented lane detection using image processing techniques. Further improvement in this project may be find distance of nearest vehicle in the lane detected, so that accidents can be avoided.

# References used -
[Lane Detection for Autonomous Vehicles using OpenCV Library](https://www.irjet.net/archives/V6/i1/IRJET-V6I1245.pdf)
