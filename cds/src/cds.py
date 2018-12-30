#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('cds')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt

class image_converter:
    def __init__(self):
        
        self.bridge = CvBridge()
        print("beforecallback")
        self.image_sub = rospy.Subscriber("Team1_image/compressed", CompressedImage, callback=self.callback, queue_size=1)
        print("aftercallback")
        rospy.Rate(10)

    def callback(self, data):
        print("callback")
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            cv2.imshow("Image window", image_np)
            cv2.waitKey(25)

        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    rospy.init_node('cds', anonymous=True)
    ic = image_converter()
    rospy.spin()
