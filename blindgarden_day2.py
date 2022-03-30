import os
import time

from math import sqrt
import numpy as np

import rospy
from clover import srv
from std_srvs.srv import Trigger
from mavros_msgs.srv import CommandBool

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from pyzbar import pyzbar

class VideoSaver:
	def __init__(self):
		self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		self.capture = cv2.VideoWriter()
		self.res = (0, 0)

	def create(self, res = (320, 240), fps = 30, output = '', time_ex = True):
		if self.capture.isOpened():
			return False

		name, _ = os.path.splitext(output)
		if name == '': name = 'video'

		if time_ex:
			addons = time.strftime("%m%d%H%M%S", time.localtime())
			name = f'{name}-{addons}.mp4'
		else: name = f'{name}.mp4'
		self.capture = cv2.VideoWriter(name, self.fourcc, fps, res)
		self.res = res
		return True

	def close(self):
		if not self.capture.isOpened():
			return False
		self.capture.release()
		return True

	def write(self, img):
		if not self.capture.isOpened():
			return False

		h, w, _ = img.shape
		if (self.res[0] != w) or (self.res[1] != h):
			self.capture.write(cv2.resize(img, self.res))
		else: self.capture.write(img)
		return True

	def __del__(self):
		self.close()

class Flight:
	def __init__(self):
		self.telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
		self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
		self.set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
		self.set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
		self.land_srv = rospy.ServiceProxy('land', Trigger)

		self.arming = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)

	def navigate_block(self, x = 0, y = 0, z = 0, yaw = float('nan'), yaw_rate = 0, \
					speed = 0, auto_arm = False, frame_id = 'body', tolerance = 0.2):
		if tolerance <= 0.05:
			print(f'Do not indicate small tolerance! (tolarnce={tolerance})')
		print(f'Moving to point x={x} y={y} z={z} ...', end = '')
		res = self.navigate(x = x, y = y, z = z, \
			yaw = yaw, yaw_rate = yaw_rate, speed = speed, \
			auto_arm = auto_arm, frame_id = frame_id)

		if not res.success:
			print('Error!')
			return res

		while not rospy.is_shutdown():
			t = self.telemetry(frame_id = 'navigate_target')
			if sqrt(t.x ** 2 + t.y ** 2 + t.z ** 2) <= tolerance:
				break
			rospy.sleep(0.1)
		print('Done!')
		return res

	def disarm(self):
		self.arming(False)

	def land(self):
		res = self.land_srv()

		print('Copter is landing...', end = '')
		if res.success:
			print('Done!')
		else: 
			print('Error!')
			return

		while self.telemetry().armed:
			rospy.sleep(0.2)

class AvoideceTask:
	def __init__(self):
		self.obstacles = []
		self.path = []

		self.size_x = 20
		self.size_y = 16
		self.radius = 0.4
		self.quad_size = 0.8

		self.navx = 0
		self.navy = 0
		self.fx = 0
		self.fy = 0
		self.v1 = 0
		self.v2 = 0

	def dst(self, xa, ya, xb, yb):
		return sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

	def dst_line2circle(self, xa, ya, xb, yb):
		"""
		return distance segment to circle with radius=r
		True if not collision or False
		"""
		xa, ya, xb, yb = xa * 0.2, ya * 0.2, xb * 0.2, yb * 0.2
		for x, y in self.obstacles:
			try:
				if xa == xb:
					if abs(self.radius) >= abs(xa - x):
						p1 = xa, y - sqrt(self.radius**2 - (xa-x)**2)
						p2 = xa, y + sqrt(self.radius**2 - (xa-x)**2)
						inp = [p1,p2]
						inp = [p for p in inp if p[1]>=min(ya,yb) and p[1]<=max(ya,yb)]
						if len(inp) > 0:
							return True
				else:
					k = (ya - yb) / (xa - xb)
					b0 = ya - k * xa

					a = k ** 2 + 1
					b = 2 * k * (b0 - y) - 2 * x
					c = (b0 - y) ** 2 + x ** 2 - self.radius ** 2
					delta = b ** 2 - 4 * a * c
					if delta >= 0: 
						p1x = (-b - sqrt(delta))/(2*a)
						p2x = (-b + sqrt(delta))/(2*a)
						p1y = k*xa + b0
						p2y = k*xb + b0
						inp = [[p1x,p1y],[p2x,p2y]]
						# select the points lie on the line segment
						inp = [p for p in inp if p[0]>=min(xa,xb) and p[0]<=max(xa,xb)]
						if len(inp) > 0:
							return True
			except: pass
		return False

	def new_obst(self):
		ln = len(self.obstacles)
		for i in range(ln - 1):
			for j in range(i + 1, ln):
				if i != j and \
						self.dst(self.obstacles[i][0], self.obstacles[i][1], \
						self.obstacles[j][0], self.obstacles[j][1]) <= self.quad_size:
					self.obstacles.append((min(self.obstacles[i][0], self.obstacles[j][0]) + \
						abs(self.obstacles[i][0] - self.obstacles[j][0]) / 2, \
						min(self.obstacles[i][1], self.obstacles[j][1]) + \
						abs(self.obstacles[i][1] - self.obstacles[j][1]) / 2))

	def check(self, x, y, reverse = False):
		if reverse:
			x, y = y, x
		cnd = [True for i in range(len(self.obstacles)) \
				if self.dst(x * 0.2, y * 0.2, self.obstacles[i][0], self.obstacles[i][1]) < self.radius]
		
		if True in cnd: return True
		return False

	def lineup(self, x_ord = True):
		first = self.navy
		diff = self.fx + abs(self.fy - self.navy)
		size = self.size_y
		if x_ord:
			first = self.navx
			diff = self.fy + abs(self.fx - self.navx)
			size = self.size_x

		for i in range(size):
			if first + i <= size and \
					self.check(first + i, diff, not x_ord) == False:
				if x_ord:
					self.v1 = first + i
					self.v2 = diff
				else:
					self.v1 = diff
					self.v2 = first + i	
				return True
			elif first - i >= 0 and \
					self.check(first - i, diff, not x_ord) == False:
				if x_ord:
					self.v1 = first - i
					self.v2 = diff
				else:
					self.v1 = diff
					self.v2 = first - i	
				return True
		return False

	def circleup(self):
		for i in range(max(self.size_x, self.size_y)):
			if self.dst_line2circle(self.fx + i, self.fy, self.v1, self.v2) == False \
					and self.fx + i <= self.size_x and \
					self.dst_line2circle(self.fx, self.fy, self.fx + i, self.fy) == False:
				self.path.append((self.fx + i, self.fy))
				self.path.append((self.v1, self.v2))
				self.fx, self.fy = self.v1, self.v2
				return True
			elif self.dst_line2circle(self.fx - i, self.fy, self.v1, self.v2) == False \
					and self.fx - i >= 0 and \
					self.dst_line2circle(self.fx, self.fy, self.fx - i, self.fy) == False:
				self.path.append((self.fx - i, self.fy))
				self.path.append((self.v1, self.v2))
				self.fx, self.fy = self.v1, self.v2
				return True
			elif self.dst_line2circle(self.fx, self.fy + i, self.v1, self.v2) == False \
					and self.fy + i <= self.size_y and \
					self.dst_line2circle(self.fx, self.fy, self.fx, self.fy + i) == False:
				self.path.append((self.fx, self.fy + i))
				self.path.append((self.v1, self.v2))
				self.fx, self.fy = self.v1, self.v2
				return True
			elif self.dst_line2circle(self.fx, self.fy - i, self.v1, self.v2) == False \
					and self.fy - i >= 0 and \
					self.dst_line2circle(self.fx, self.fy, self.fx, self.fy - i) == False:
				self.path.append((self.fx, self.fy - i))
				self.path.append((self.v1, self.v2))
				self.fx, self.fy = self.v1, self.v2
				return True
		return False

	def vectorup(self, x_ord = True):
		self.v1, self.v2 = self.navx, self.navy
		if not self.circleup():
			if x_ord:
				for i in range(12):
					self.v1, self.v2 = self.navx + i, self.navy
					if self.circleup(): break
					self.v1, self.v2 = self.navx - i, self.navy
					if self.circleup(): break
			else:
				for i in range(12):
					self.v1, self.v2 = self.navx, self.navy + i
					if self.circleup(): break
					self.v1, self.v2 = self.navx, self.navy - i
					if self.circleup(): break
		else: 
			self.fx = self.navx
			self.fy = self.navy

	def gen_path(self, start, finish, deepth = 15):
		self.path = []
		self.navx, self.navy = round(finish[0] / 0.2, 0), round(finish[1] / 0.2, 0)
		self.fx, self.fy = round(start[0] / 0.2, 0), round(start[1] / 0.2, 0)

		while ((self.fx != self.navx) or (self.fy != self.navy)) \
				and deepth > 0:
			deepth -= 1
			if self.fx == self.navx:
				self.vectorup(False)
			elif self.fy == self.navy:
				self.vectorup()
			else:
				if self.navx < self.navy:
					self.lineup()
				else: self.lineup(False)
				self.circleup()

class Task:
	def __init__(self, record = False, output_file = ''):
		self.flight = Flight()

		self.record = record
		if self.record:
			self.video = VideoSaver()
			self.video.create(output = output_file)

		self.QR_detect = False
		self.nav_area = []

		self.Line_detect = False

		self.cx = 0
		self.cy = 0

		self.cv_debug = rospy.Publisher('CV_debug', Image, queue_size=1)

		self.bridge = CvBridge()
		rospy.Subscriber('main_camera/image_raw', \
				Image, self.image_cb)

	def normilize(self, x, y):
		dst = sqrt(x ** 2 + y ** 2)
		return x / dst, y / dst

	def image_cb(self, msg):
		try:
			image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
		except CvBridgeError as e: print(e)

		if self.record: self.video.write(image)

		if self.QR_detect:
			image = self.qr_code(image)

		if self.Line_detect:
			image = self.line_detect(image)

		try:
			self.cv_debug.publish(self.bridge.cv2_to_imgmsg(image, encoding = 'bgr8'))
		except CvBridgeError as e: print(e)

	def qr_code(self, image):
		boxes = pyzbar.decode(image)
		if len(boxes) == 0:
			return image
		
		for box in boxes:
			data = box.data.decode("utf-8").split('\n')
			if len(data) != 1:
				print('Error data format in Qr-code, skipping!')
				return image

			image = cv2.rectangle(image, (box.rect.left, box.rect.top), \
					(box.rect.width + box.rect.left, box.rect.height + box.rect.top), (0, 255, 0), 3)
			
			if len(self.nav_area) != 0:
				return image

			self.nav_area = data[0].split(' ')
			break
		return image

	def line_detect(self, image):
		hsv = cv2.cvtColor(cv2.GaussianBlur(image,(5,5), 0), \
			cv2.COLOR_BGR2HSV)
		bin = cv2.inRange(hsv, (0,46,117), (61, 159, 255))

		bin = bin[35:(240 // 2) + 100, :]
		kernel = np.ones((1, 1), 'uint8')
		bin = cv2.erode(bin, kernel)
		contours_blk, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		if len(contours_blk) > 0:
			cnt = max(contours_blk, key = cv2.contourArea)
			if cv2.contourArea(cnt) > 300:
				rect = cv2.minAreaRect(cnt)

				box = cv2.boxPoints(rect)
				box = np.int0(box)
				for i in range(4):
					box[i][1] += 35
				image = cv2.drawContours(image,[box],0,(0,0,255),2)


				(x_min, y_min), (w_min, h_min), angle = rect
				print(round(angle, 2), w_min, h_min)
				if angle < -45:
					angle = 90 + angle
				if w_min < h_min and angle > 0:
					angle = (90 - angle) * -1
				if w_min > h_min and angle < 0:
					angle = 90 + angle

				center = image.shape[1] / 2
				error = x_min - center

				print(round(angle, 2), error)
				self.flight.set_velocity(vx=0.1, vy=error*(-0.006), vz=0, yaw=float('nan'), yaw_rate=angle*(-0.006), frame_id='body')
		return image


rospy.init_node('flight_node')

task = Task(record = True, output_file = 'green.mp4')

height = 1.0
height_low_detect = 1.0
task.flight.navigate_block(z = 0.5, frame_id = 'body', speed = 0.3, tolerance = 0.25, auto_arm = True)

task.QR_detect = True
rospy.sleep(4)

if len(task.nav_area) == 0:
	#t = task.flight.telemetry(frame_id = 'aruco_map')
	#task.flight.navigate_block(x = t.x, y = t.y, z = 0., frame_id = 'aruco_map', speed = 0.3, tolerance = 0.15)
	#rospy.sleep(2)

	if len(task.nav_area) == 0:
		task.flight.land()
		print('No qr-code detect')
		exit()
task.QR_detect = False

#task.flight.navigate_block(z = 1.0, frame_id = 'body', speed = 0.3, tolerance = 0.15)
print(f'Qr-code info: x={task.nav_area[0]} y={task.nav_area[1]}')

task.flight.navigate_block(x = float(task.nav_area[0]), y = float(task.nav_area[1]), z = height_low_detect, tolerance = 0.15, speed = 0.3, frame_id = 'aruco_map')
task.Line_detect = True
rospy.sleep(50)
task.flight.land()
