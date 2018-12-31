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
from lane_detector import lane_detect

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("Team1_image/compressed", CompressedImage, callback=self.callback, queue_size=1)
        rospy.Rate(10)

    def callback(self, data):
        # print("callback")
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # NOTE: image_np.shape = (240,320,3)
            cv2.imshow("Image window", image_np)
            cv2.waitKey(1)
            left_fit, right_fit, out_img = lane_detect(image_np)

            cv2.imshow("Detect_lane", out_img)
            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    rospy.init_node('cds', anonymous=True)
    cv2.namedWindow("houghlines")
    def nothing():
        pass
    cv2.createTrackbar("rho", "houghlines",2,255,nothing)
    cv2.createTrackbar("theta", "houghlines",180,255,nothing)
    cv2.createTrackbar("minLine", "houghlines",78,255,nothing)
    cv2.createTrackbar("maxGap", "houghlines",10,255,nothing)
    cv2.waitKey(1)
    ic = image_converter()
    rospy.spin()
