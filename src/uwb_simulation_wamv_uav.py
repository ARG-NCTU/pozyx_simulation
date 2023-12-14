#!/usr/bin/env python

import numpy as np
import rospy
import scipy
import tf
from geometry_msgs.msg import PoseStamped
from scipy.optimize import least_squares
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray


class uwb_ranging(object):
    def __init__(self):
        rospy.init_node("uwb_simulation", anonymous=True)
        self.tag_frame = rospy.get_param("~tag_frame", "drone/uwb/0")
        self.robot_name = rospy.get_param("~robot_name", "wamv")
        self.robot_base_frame = rospy.get_param("~robot_base_frame", "base_link")
        
        self.robot_base_frame = self.robot_name + "/" + self.robot_base_frame
        
        self.robot_pose_tf = [0.0, 0.0, 0.0]

        self.counter = 0
        self.residual = 0.0

        # get uwb anchors position
        self.listener = tf.TransformListener()
        self.anchor_poses = []
        self.anchor_poses = self.get_anchors_pos()
        self.num_anchors = len(self.anchor_poses)

        # distances are publishing with uwb_data_distance
        self.uwb_distances = Float64MultiArray()
        self.uwb_distances.data = [0.0 for _ in range(self.num_anchors)]
        self.pub_uwb_distances = rospy.Publisher("distance", Float64MultiArray, queue_size=0)

        self.estimated_pose = PoseStamped()
        self.pub_estimated_pose = rospy.Publisher("pose/estimated", PoseStamped, queue_size=0)

        self.ground_truth = PoseStamped()
        self.pub_ground_truth = rospy.Publisher("pose/ground_truth", PoseStamped, queue_size=0)
        

        # start the publish uwb data
        rospy.Timer(rospy.Duration(1 / 20.0), self.get_robot_pose)
        rospy.Timer(rospy.Duration(1 / 20.0), self.uwb_simulate)

    def get_anchors_pos(self, try_time=0):
        if try_time > 100:
            return
        rospy.sleep(5)
        max_anchor = 10
        uwb_id = self.robot_name + "/uwb/"

        for i in range(max_anchor):
            try:
                rospy.sleep(0.3)
                (trans, rot) = self.listener.lookupTransform(self.robot_base_frame, uwb_id + str(i), rospy.Time(0))
                self.anchor_poses.append(trans)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("[UWB Simulation]: " + "There is not found " + self.robot_base_frame + " to "+ uwb_id + str(i) + " anchor.")

        if self.anchor_poses == []:
            rospy.logwarn("[UWB Simulation]: " + "There is not found any anchors. Function is working again.")
            rospy.logwarn("[UWB Simulation]: " + "Please check the anchor frame name: " + uwb_id + "0 ~ " + uwb_id + str(max_anchor) + " is exist.")
            self.get_anchors_pos(try_time=try_time + 1)
        else:
            rospy.loginfo("[UWB Simulation]: " + "UWB Anchor List:\nWarning : uint is mm \n" + str(self.anchor_poses))

        return self.anchor_poses

    def cal_residuals(self, guess):
        x0, y0, z0, r = guess

        residuals = [0.0 for _ in range(self.num_anchors)]
        for i in range(self.num_anchors):
            residual = np.square(x0 - self.anchor_poses[i][0]) + np.square(y0 - self.anchor_poses[i][1]) + np.square(z0 - self.anchor_poses[i][2]) - np.square(r - self.uwb_distances.data[i])
            residuals[i] = residual
        residuals = np.array(residuals).flatten()
        return residuals

    def uwb_simulate(self, e):
        # Set ground truth
        self.ground_truth.pose.position.x = self.robot_pose_tf[0]
        self.ground_truth.pose.position.y = self.robot_pose_tf[1]
        self.ground_truth.pose.position.z = self.robot_pose_tf[2]

        for i in range(self.num_anchors):
            # calculate distance uwb to robot for all anchors
            dist = self.calculate_distance(self.anchor_poses[i])
            self.uwb_distances.data[i] = dist
        try:
            result = least_squares(self.cal_residuals, (self.robot_pose_tf[0], self.robot_pose_tf[1], self.robot_pose_tf[2], 0))
            self.estimated_pose.pose.position.x, self.estimated_pose.pose.position.y, self.estimated_pose.pose.position.z, self.residual = result.x
            # rospy.loginfo_throttle(1, result.x)
            self.publish()
        except ValueError:
            rospy.loginfo_once("ValueError")

    def calculate_distance(self, uwb_pose):
        # describe 2 points
        p1 = np.array(uwb_pose)
        p2 = np.array(self.robot_pose_tf)

        # difference between robot and uwb distance
        uwb_dist = np.sqrt(np.sum((p1 - p2) ** 2, axis=0))
        
        # Stddev model
        stddev_slope = 0.03824
        stddev_intercept = 0.11578
        ## This is from EE6F test result
        stddev_slope = 0.00135289
        stddev_intercept = 0.0945902
        
        stddev_slope = 0.0
        stddev_intercept = 0.0
        # print uwb_dist and normal size
        uwb_dist = np.random.normal(uwb_dist, stddev_slope * uwb_dist + stddev_intercept, (1,))
        return uwb_dist[0]

    def publish(self):
        now = rospy.Time.now()
        
        self.estimated_pose.header.stamp = now
        self.estimated_pose.header.frame_id = self.robot_base_frame
        self.estimated_pose.pose.orientation.w = 1.0
        self.pub_estimated_pose.publish(self.estimated_pose)

        
        self.ground_truth.header.stamp = now
        self.ground_truth.header.frame_id = self.robot_base_frame
        self.ground_truth.pose.orientation.w = 1.0
        self.pub_ground_truth.publish(self.ground_truth)
        
        self.uwb_distances.data = [int(i * 1000) for i in self.uwb_distances.data]
        self.pub_uwb_distances.publish(self.uwb_distances)


    def get_robot_pose(self, e):
        try:
            (trans, rot) = self.listener.lookupTransform(self.robot_base_frame, self.tag_frame, rospy.Time(0))
            self.robot_pose_tf = trans
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass


if __name__ == "__main__":
    uwb_simulation = uwb_ranging()
    rospy.spin()
