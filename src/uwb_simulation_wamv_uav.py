#!/usr/bin/env python

import rospy
import tf 
import numpy as np
import scipy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from scipy.optimize import least_squares

from visualization_msgs.msg import Marker, MarkerArray

class uwb_ranging(object):
    def __init__(self):
        self.robot_pose_tf = [0.0, 0.0, 0.0]

        self.counter = 0 
        self.residual = 0.0

        #get uwb anchors position
        self.listener = tf.TransformListener()
        self.anchor_poses = []
        self.anchor_poses = self.get_anchors_pos()
        self.num_anchors = len(self.anchor_poses)

        #distances are publishing with uwb_data_distance
        self.uwb_distances = Float32MultiArray()
        self.uwb_distances.data = [0.0 for i in range(self.num_anchors)]
        self.pub_uwb_distances = rospy.Publisher('~/distance', Float32MultiArray, queue_size=0)

        self.uwb_robot_pose = PoseStamped()
        self.pub_robot_pose = rospy.Publisher('~/robot_pose', PoseStamped, queue_size=0)

        self.pub_points = rospy.Publisher("anchor_viz", MarkerArray, queue_size=0)
        self.markers = MarkerArray()
        self.markers.markers = []
        self.publish_markers()

        #start the publish uwb data
        rospy.Timer(rospy.Duration(1/20.0), self.get_robot_pose)
        rospy.Timer(rospy.Duration(1/100.0), self.uwb_simulate)

    def get_anchors_pos(self):
        max_anchor = 10 
        uwb_id = 'wamv/uwb/'

        for i in range(max_anchor):
            try:
                rospy.sleep(0.3)
                (trans, rot) = self.listener.lookupTransform('map', uwb_id+str(i), rospy.Time(0))
                self.anchor_poses.append(trans)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                break

        if self.anchor_poses == []:
            rospy.logwarn("There is not found any anchors. Function is working again.")    
            self.get_anchors_pos()
        else:
            print(self.anchor_poses)
            rospy.loginfo("UWB Anchor List:\nWarning : uint is mm \n" + str(self.anchor_poses))

        return self.anchor_poses

    def cal_residuals(self, guess):
        x0, y0, z0, r = guess

        residuals = [0.0 for i in range(self.num_anchors)]
        for i in range(self.num_anchors):
            residual = np.square(x0 - self.anchor_poses[i][0]) \
                     + np.square(y0 - self.anchor_poses[i][1]) \
                     + np.square(z0 - self.anchor_poses[i][2]) \
                     - np.square(r - self.uwb_distances.data[i])
            residuals[i] = residual
        residuals = np.array(residuals).flatten()
        return residuals
        

    def uwb_simulate(self, e):
        for i in range(self.num_anchors):
            #calculate distance uwb to robot for all anchors 
            dist = self.calculate_distance(self.anchor_poses[i])
            self.uwb_distances.data[i] = dist

        result = least_squares(self.cal_residuals, (self.robot_pose_tf[0], self.robot_pose_tf[1], self.robot_pose_tf[2], 0))
        self.uwb_robot_pose.pose.position.x, self.uwb_robot_pose.pose.position.y, self.uwb_robot_pose.pose.position.z, self.residual = result.x
        rospy.loginfo_throttle(1, result.x)
        self.publish()

    def calculate_distance(self, uwb_pose):
        #describe 2 points
        p1 = np.array(uwb_pose)
        p2 = np.array(self.robot_pose_tf)

        #difference between robot and uwb distance
        uwb_dist = np.sum((p1-p2)**2, axis=0)

        #add noise 
        uwb_dist = np.random.normal(uwb_dist, uwb_dist*0.0006, 1)  
        return np.sqrt(uwb_dist)[0]

        

    def publish(self):
        #uwb message type is a special message so that firstly describe this message 
        self.uwb_robot_pose.header.stamp = rospy.Time.now()
        self.uwb_robot_pose.header.frame_id = 'map'
        self.uwb_robot_pose.pose.orientation.w = 1.0
        self.pub_robot_pose.publish(self.uwb_robot_pose)

        self.pub_uwb_distances.publish(self.uwb_distances)
        self.publish_markers()


    def publish_markers(self):
        now = rospy.Time.now()
        for i, pose in enumerate(self.anchor_poses):
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = 'map'
            marker.id = i
            marker.type = marker.CUBE
            marker.action = marker.ADD
            marker.pose.position.x = pose[0]
            marker.pose.position.y = pose[1]
            marker.pose.position.z = pose[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1
            marker.color.g = 0
            marker.color.b = 0
            marker.lifetime = rospy.Duration.from_sec(0.0)
            self.markers.markers.append(marker)
        self.pub_points.publish(self.markers)

    def get_robot_pose(self, e):
        try:
            (trans, rot) = self.listener.lookupTransform('map', 'fcu', rospy.Time(0))
            self.robot_pose_tf = trans
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass
        

if __name__ == "__main__":
    rospy.init_node('uwb_simulation', anonymous=True)

    uwb_simulation = uwb_ranging()
    rospy.spin()

