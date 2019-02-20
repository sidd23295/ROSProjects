#!/usr/bin/env python

# Assignment 4 - Siddharth Chamarthy
# UNI - sc4253

import numpy
import sys
import math
import random

from geometry_msgs.msg import *
import moveit_msgs.msg
import moveit_msgs.srv
import rospy
import tf
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import *

def convert_message(M):
	# gathers the translation part from the message
	t = tf.transformations.translation_matrix((M.translation.x,
												  M.translation.y,
												  M.translation.z))
	# gathers the rotation part from the message
	r = tf.transformations.quaternion_matrix((M.rotation.x,
												M.rotation.y,
												M.rotation.z,
												M.rotation.w))
	# gets the net transformation
	M = numpy.dot(t,r)
	return M

def convert_to_message(T):
	t = geometry_msgs.msg.Pose()
	position = tf.transformations.translation_from_matrix(T)
	orientation = tf.transformations.quaternion_from_matrix(T)
	t.position.x = position[0]
	t.position.y = position[1]
	t.position.z = position[2]
	t.orientation.x = orientation[0]
	t.orientation.y = orientation[1]
	t.orientation.z = orientation[2]
	t.orientation.w = orientation[3]        
	return t


class MoveArm(object):

	def __init__(self):


		# Wait for moveit IK service
		rospy.wait_for_service("compute_ik")
		self.ik_service = rospy.ServiceProxy('compute_ik',  moveit_msgs.srv.GetPositionIK)
		print "IK service ready"

		# Wait for validity check service
		rospy.wait_for_service("check_state_validity")
		self.state_valid_service = rospy.ServiceProxy('check_state_validity',  
													  moveit_msgs.srv.GetStateValidity)
		print "State validity service ready"

		self.n_j = 7

		self.names = ["lwr_arm_0_joint","lwr_arm_1_joint","lwr_arm_2_joint","lwr_arm_3_joint","lwr_arm_4_joint","lwr_arm_5_joint","lwr_arm_6_joint"]

		self.current_joint_state = JointState()
		# MoveIt parameter
		self.group_name = "lwr_arm"

		rospy.Subscriber("/joint_states", JointState, self.j_callback)

		rospy.Subscriber("/motion_planning_goal", Transform, self.plan_callback)

		self.pub = rospy.Publisher("/joint_trajectory", trajectory_msgs.msg.JointTrajectory, queue_size = 1)

	def IK(self, T_goal):
		req = moveit_msgs.srv.GetPositionIKRequest()
		req.ik_request.group_name = self.group_name
		req.ik_request.robot_state = moveit_msgs.msg.RobotState()
		req.ik_request.robot_state.joint_state.name = ["lwr_arm_0_joint",
													   "lwr_arm_1_joint",
													   "lwr_arm_2_joint",
													   "lwr_arm_3_joint",
													   "lwr_arm_4_joint",
													   "lwr_arm_5_joint",
													   "lwr_arm_6_joint"]
		req.ik_request.robot_state.joint_state.position = numpy.zeros(7)
		req.ik_request.robot_state.joint_state.velocity = numpy.zeros(7)
		req.ik_request.robot_state.joint_state.effort = numpy.zeros(7)
		req.ik_request.robot_state.joint_state.header.stamp = rospy.get_rostime()
		req.ik_request.avoid_collisions = True
		req.ik_request.pose_stamped = geometry_msgs.msg.PoseStamped()
		req.ik_request.pose_stamped.header.frame_id = "world_link"
		req.ik_request.pose_stamped.header.stamp = rospy.get_rostime()
		req.ik_request.pose_stamped.pose = convert_to_message(T_goal)
		req.ik_request.timeout = rospy.Duration(3.0)
		res = self.ik_service(req)
		q = []
		if res.error_code.val == res.error_code.SUCCESS:
			q = res.solution.joint_state.position
		return q

	def is_state_valid(self, q):
		req = moveit_msgs.srv.GetStateValidityRequest()
		req.group_name = self.group_name
		req.robot_state = moveit_msgs.msg.RobotState()
		req.robot_state.joint_state.name = ["lwr_arm_0_joint",
											"lwr_arm_1_joint",
											"lwr_arm_2_joint",
											"lwr_arm_3_joint",
											"lwr_arm_4_joint",
											"lwr_arm_5_joint",
											"lwr_arm_6_joint"]
		req.robot_state.joint_state.position = q
		req.robot_state.joint_state.velocity = numpy.zeros(7)
		req.robot_state.joint_state.effort = numpy.zeros(7)
		req.robot_state.joint_state.header.stamp = rospy.get_rostime()
		res = self.state_valid_service(req)
		return res.valid

	def get_values(self, data, j_name):
		if j_name not in data.name:
			return
		k = data.name.index(j_name)
		return data.position[k]

	# this function is used to find the vector between two points
	def vector(self, q1, q2):
		q = numpy.subtract(q1,q2)
		return q

	# this function is used to find the unit vector between two points
	def unit_vector_convertor(self, q1, q2):
		v = self.vector(q1, q2)
		u = v / numpy.linalg.norm(v)
		return u

	# this function is used to find the distance between the two points in joint space
	def norm(self, q1, q2):
		v = self.vector(q1, q2)
		d = numpy.linalg.norm(v)
		return d

	# This function does the main collision check in the rrt algorithm
	def collision_check(self, p_close, p_target):
		q_s = [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05]
		M = []
		v = self.vector(p_target, p_close)
		u = abs(v)
		# gets number of points for a given sample size
		s = numpy.true_divide(u,q_s)
		t = numpy.ceil(s)
		# gets the maximum of the number of points
		n_points = max(t)
		# divides the vector between the close point and the target point by the 
		# total number of points to give step length
		p = numpy.true_divide(v, n_points)
		# gets the maximum length from the closest point
		l = numpy.multiply(p_close, n_points)
		# gets the vector between target and closest point
		m = numpy.subtract(l, p_target)
		# gets base vector from this computation 
		b = numpy.true_divide(m, n_points)
		for i in range(int(n_points)):
			# gets all the points between the target point and close point
			M.append((p*(i+1)+b))
		# checks if all the lists in the given array of lists are not colliding
		for t in M:
			if self.is_state_valid(t) == False:
				return False
		return True

	def close(self, rrt_tree, q):
		d = []
		# Since the rrt is stored in a list of dictionary objects we need to enumerate it as long as there is an object in the rrt_tree
		for k,pos in enumerate(p["position"] for p in rrt_tree):
			# All the distances between the current random q and the nodes in the tree are stored in a list
			d.append(self.norm(pos,q))
		# the index of the smallest element is stored in a variable
		min_index = d.index(min(d))
		# the position of the closest point on the tree is returned
		p_close = rrt_tree[min_index].get("position")
		return min_index, p_close


	def rrt_function(self, s1, s2):
		# creation of a dictionary that stores position and index of the specific source 
		dictionary = {"position": s1, "source_node" : -1}
		# object of the dictionary acts like a list
		rrt = []
		rrt.append(dictionary.copy())
		# copies all the dictionary contents to the rrt tree object
		while True:
			# loop to find random joint angles between -pi and +pi
			q_rand = []
			for k in range(7):
				N = random.uniform(-1*math.pi,math.pi)
				q_rand.append(N)

			# This function call is used to find the closest point in the rrt list to the randomly generated joint angles
			min_d_index, p_close = self.close(rrt,q_rand)

			# This code uses the close point from the rrt and finds a point 0.3 units away from it in the direction of the random point
			p_target = self.unit_vector_convertor(q_rand, p_close)
			p_target = numpy.multiply(p_target,0.3)
			p_target = numpy.add(p_target,p_close)

			# This is the beginning of the main algorithm for the rrt collision check
			# primary check: if there is any obstacle between target point and closest point
			if self.collision_check(p_close,p_target) == True:
				dictionary.update({"position": p_target})
				dictionary.update({"source_node": min_d_index})
				rrt.append(dictionary.copy())
				print len(rrt)
				# Secondary check: if there is any obstacle between target point and goal point
				if self.collision_check(p_target, s2) == True:
					source_node = len(rrt)-1
					dictionary.update({"source_node": source_node})
					dictionary.update({"position": s2})
					rrt.append(dictionary.copy())
					print 'goal has been reached'
					break

		# start retracing path backwards
		q_list = [s2]
		source_node = rrt[-1].get("source_node")

		while True:
			pos_node = rrt[source_node].get("position")
			q_list.insert(0, pos_node)
			if source_node <=0:
				break
			else:
				source_node = rrt[source_node].get("source_node")

		# shortcutting by just checking any two points on the q_list		
		copy = []
		copy.append(q_list[0])
		u = 0
		for u in range(len(q_list)-2):
			if self.collision_check(q_list[u], q_list[u+2]) == False:
				copy.append(q_list[u])
		copy.append(q_list[-1])
		q_list = copy

		# Resampling the path so that continuous motion is assured
		q_new = []
		q_new.append(list(q_list[0]))
		c = 1
		z = 0
		while c<(len(q_list)):
			v = self.vector(q_list[c],q_list[z])
			norm = numpy.linalg.norm(v)
			numb_points = norm/0.5
			ceil_n_p = int(numpy.ceil(numb_points))
			if ceil_n_p == 0:
				q_new.append(q_list[z])
				q_new.append(q_list[c])
			else:
				segment_steps = numpy.divide(v,ceil_n_p)
				j = 0
				# loop for appending the new set of points with steps 
				for j in range(ceil_n_p):
					q_new.append(q_list[z]+((j+1)*segment_steps))
			z = z+1
			c = c+1
		return q_new


	def plan_callback(self, data):
		# everytime command is given this function gets a transform from the message 
		# and grabs the transformation from it
		t_g = convert_message(data)

		# the list of starting joint values taken from the current joint state
		q_st = []
		for t in range(0, self.n_j):
			p = self.get_values(self.current_joint_state, self.names[t])
			q_st.append(p)

		# This function solves the IK for the given message and returns joint values
		q_g = self.IK(t_g)
		# print q_g
		if len(q_g) == 0:
			print 'IK failed'
			return
		print 'IK solved'

		# converts the given list of joint angles into an array for use in the RRT function
		q_st = numpy.array(q_st)
		# solves the motion planning problem and returns list of joint values which belong to the trajectory
		q_l = self.rrt_function(q_st, q_g)
		# creates the joint trajectory for list of joint angles
		joint_tra = self.trajectory_creator(q_l)
		if not joint_tra.points:
			print 'failed'
		else:
			# publishes the joint trajectory
			self.pub.publish(joint_tra)
			
	def trajectory_creator(self, path):
		t = trajectory_msgs.msg.JointTrajectory()
		for i in range(0, len(path)):
			p = trajectory_msgs.msg.JointTrajectoryPoint()
			# stores the positions as a list 
			p.positions = list(path[i])
			p.velocities = []
			p.accelerations = []
			t.points.append(p)
		t.joint_names = self.names
		return t

	def j_callback(self, joint_states):
		self.current_joint_state = joint_states

if __name__ == '__main__':
	rospy.init_node('move_arm', anonymous=True)
	ma = MoveArm()
	rospy.spin()

