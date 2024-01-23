#!/usr/bin/env python

import numpy as np
import rospy
import scipy
import tf
from geometry_msgs.msg import PoseStamped
from scipy.optimize import least_squares
from std_msgs.msg import Float64MultiArray


class uwb_ranging(object):
    def __init__(self):
        rospy.init_node("uwb_simulation")
        self.anchor_frame_prefix = rospy.get_param("~anchor_frame_prefix", "wamv/uwb")
        self.tag_frame = rospy.get_param("~tag_frame", "drone/uwb/0")
        self.robot_name = rospy.get_param("~robot_name", "wamv")
        self.robot_base_frame_suffix = rospy.get_param("~robot_base_frame", "base_link")
        self.anchor_num = rospy.get_param("~anchor_num", 6)

        self.robot_base_frame = (
            self.robot_name
            + ("/" if not self.robot_base_frame_suffix.startswith("/") else "")
            + self.robot_base_frame_suffix
        )

        # get uwb anchors position
        self.listener = tf.TransformListener()
        self.anchor_poses = np.zeros((self.anchor_num, 3))
        self.anchor_poses = self.get_anchors_pos(self.anchor_num)
        rospy.loginfo("\nAnchor poses: \n{}\n".format(self.anchor_poses))

        # distances are publishing with uwb_data_distance
        self.uwb_distances = Float64MultiArray()
        self.uwb_distances.data = [0.0 for _ in range(self.anchor_num)]
        self.pub_uwb_distances = rospy.Publisher("distances", Float64MultiArray, queue_size=0)

        self.ground_truth = PoseStamped()
        self.pub_ground_truth = rospy.Publisher("pose/ground_truth", PoseStamped, queue_size=0)

        self.robot_pose_tf = np.zeros((3,))

        # start the publish uwb data
        rospy.Timer(rospy.Duration(1 / 50.0), self.get_robot_pose)
        rospy.Timer(rospy.Duration(1 / 20.0), self.uwb_simulate)

    def get_anchors_pos(self, anchor_num):
        anchor_poses = np.zeros((anchor_num, 3))
        success = np.zeros(anchor_num)
        while True:
            rospy.sleep(0.5)
            rospy.loginfo("Waiting for tf")
            for i in range(anchor_num):
                anchor_frame = (
                    self.anchor_frame_prefix + ("/" if not self.anchor_frame_prefix.endswith("/") else "") + str(i)
                )
                rospy.loginfo("Look up {} to {}".format(self.robot_base_frame, anchor_frame))
                try:
                    self.listener.waitForTransform(
                        self.robot_base_frame, anchor_frame, rospy.Time(0), rospy.Duration(0.5)
                    )
                    trans, rot = self.listener.lookupTransform(self.robot_base_frame, anchor_frame, rospy.Time(0))
                    anchor_poses[i] = np.array(trans)
                    success[i] = 1
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logerr_once("Anchor tf not ready yet, retrying for anchor %d" % i)
            if np.all(success):
                break
        return anchor_poses

    def get_robot_pose(self, e):
        try:
            (trans, rot) = self.listener.lookupTransform(self.robot_base_frame, self.tag_frame, rospy.Time(0))
            self.robot_pose_tf = trans
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Robot tf not ready yet")

    def uwb_simulate(self, e):
        # Set ground truth
        self.ground_truth.pose.position.x = self.robot_pose_tf[0]
        self.ground_truth.pose.position.y = self.robot_pose_tf[1]
        self.ground_truth.pose.position.z = self.robot_pose_tf[2]

        for i in range(self.anchor_num):
            # calculate distance uwb to robot for all anchors
            dist = self.calculate_distance(self.anchor_poses[i])
            self.uwb_distances.data[i] = dist
        try:
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

        self.ground_truth.header.stamp = now
        self.ground_truth.header.frame_id = self.robot_base_frame
        self.ground_truth.pose.orientation.w = 1.0
        self.pub_ground_truth.publish(self.ground_truth)

        self.uwb_distances.data = [int(i * 1000) for i in self.uwb_distances.data]
        self.pub_uwb_distances.publish(self.uwb_distances)


if __name__ == "__main__":
    uwb_simulation = uwb_ranging()
    rospy.spin()
