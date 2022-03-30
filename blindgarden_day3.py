
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

# Класс для записи видео с камеры клевера в файл
# с уникальным названием
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

	# метод для записи кадра
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

# Класс для реализации управления коптером
class Flight:
	def __init__(self):
		self.telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
		self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
		self.set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
		self.set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
		self.land_srv = rospy.ServiceProxy('land', Trigger)

		self.arming = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)

	# полет с ожиданием по координатам
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

	# дизарм
	def disarm(self):
		self.arming(False)

	# посадка
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

# Основной класс, реализует распознавание всего, что нужно
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
		self.error_time = 0
		self.error = False

		self.line_end = False
		self.line_end_time = 0

		self.Oil_detect = False
		self.error_cx = 0
		self.error_cy = 0
		self.oil_time = 0
		self.oil = False
		self.oil_area = 0.0

		self.cx = 0
		self.cy = 0

		# для вывода кадров
		self.cv_debug = rospy.Publisher('CV_debug', Image, queue_size=1)
		self.defect_topic = rospy.Publisher('/defect_detect', Image, queue_size=1)
		self.oil_topic = rospy.Publisher('/oil_detect', Image, queue_size=1)

		self.bridge = CvBridge()
		rospy.Subscriber('main_camera/image_raw_throttled', \
				Image, self.image_cb)

	def normilize(self, x, y):
		dst = sqrt(x ** 2 + y ** 2)
		return x / dst, y / dst

	# callback функция для приема кадров
	def image_cb(self, msg):
		try:
			image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
		except CvBridgeError as e: print(e)

		if self.record: self.video.write(image)
		cv_debug = image.copy()
		defect_image = image.copy()
		oil_image = image.copy()


		if self.QR_detect:
			cv_debug = self.qr_code(image)

		if self.Line_detect:
			cv_debug, defect_image = self.line_detect(image)

		if self.Oil_detect:
			oil_image = self.oil_detect(image)

		try:
			self.cv_debug.publish(self.bridge.cv2_to_imgmsg(cv_debug, encoding = 'bgr8'))
			self.defect_topic.publish(self.bridge.cv2_to_imgmsg(defect_image, encoding = 'bgr8'))
			self.oil_topic.publish(self.bridge.cv2_to_imgmsg(oil_image, encoding = 'bgr8'))
		except CvBridgeError as e: print(e)

	# распознавание QR-кодов
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

	# метод для нормализации угла поворота линии в кадре
	def ang_normilize(self, w_min, h_min, ang):
		if ang < -45:
				ang = 90 + ang
		if w_min < h_min and ang > 0:
				ang = (90 - ang) * -1
		if w_min > h_min and ang < 0:
				ang = 90 + ang
		return ang

	# распознавание линии и повреждений
	def line_detect(self, image):
		image_draw = image.copy()
		defect_image = image.copy()
		hh, ww, _ = image.shape
		vx_speed = 0.1

		blur = cv2.GaussianBlur(image,(5,5), 0)
		hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
		bin = cv2.inRange(hsv, (23, 48, 141), (52, 180, 255))

		bin = bin[(hh // 2) - 60:(hh // 2) + 90, :]
		image_roi = image[(hh // 2) - 60:(hh // 2) + 90, :]
		kernel = np.ones((5,5),np.uint8)
		bin = cv2.erode(bin, kernel)
		bin = cv2.dilate(bin, kernel)

		#cv2.imshow('bin_line', bin)
		#cv2.waitKey(1)

		contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		coords = []
		max_area_y = 0
		max_area_rect = 0.0; max_rect = None
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if cv2.contourArea(cnt) > 300:
				rect = cv2.minAreaRect(cnt)

				(x_min, y_min), (w_min, h_min), angle = rect

				ratio = max(h_min / w_min, w_min / h_min)
				if abs(1.0 - ratio) > 1.0:
					box = cv2.boxPoints(rect)
					box = np.int0(box)
					for i in range(4): box[i][1] += (hh // 2) - 60

					b_min = min(box, key = lambda x: x[1])
					b_max = max(box, key = lambda x: x[1])
					coords.append([b_min, b_max])

					#if max_area_rect < area:
					if max_area_y < y_min:
						max_area_y = y_min
						#max_area_rect = area
						max_rect = [(x_min, y_min), (w_min, h_min), angle]

					image_draw = cv2.circle(image_draw, (int(x_min), int(y_min + (hh // 2) - 60)), 5, (127, 0, 127), -1)
					image_draw = cv2.drawContours(image_draw,[box],0,(127,0,255),3)
		if len(coords) > 1:
			coords = sorted(coords, key = lambda x: x[0][1])
			image_draw = cv2.circle(image_draw, (int(coords[0][1][0]), int(coords[0][1][1])), 5, (127, 255, 0), -1)
			image_draw = cv2.circle(image_draw, (int(coords[-1][0][0]), int(coords[-1][0][1])), 5, (127, 255, 0), -1)
			y_min = int(coords[0][1][1])
			x_min = int(coords[0][1][0])
			y_max = int(coords[-1][0][1])

			#print(y_min, y_max)

			mask_search = image[y_min - 30:y_max + 30, :, 1].copy()
			mask_search = cv2.inRange(mask_search, 0, 130)
			kernel = np.ones((3,3),np.uint8)
			mask_search = cv2.erode(mask_search, kernel)

			#cv2.imshow('bin', mask_search)
			#cv2.waitKey(1)

			contours, _ = cv2.findContours(mask_search, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			points = []
			y_min_local = y_min - (y_min - 30)
			for cnt in contours:
				if cv2.contourArea(cnt) > 10 and cv2.contourArea(cnt) < 750:
					x, y, w, h = cv2.boundingRect(cnt)
					points.append([sqrt((y_min_local - y) ** 2 + (x_min - x) ** 2), x, y, w, h, cnt])
			points = sorted(points, key = lambda x: x[0])
			defect_image[y_min - 30:y_max + 30, :] = cv2.drawContours(defect_image[y_min - 30:y_max + 30, :],[points[0][5]],0,(180,105,255),3)
			image_draw[y_min - 30:y_max + 30, :] = cv2.rectangle(image_draw[y_min - 30:y_max + 30, :], \
					(points[0][1], points[0][2]), (points[0][1] + points[0][3], points[0][2] + points[0][4]), (180, 150, 255), 3)
			if self.error_time == 0: 
				self.error_time = time.time()
			elif (time.time() - self.error_time) >= 0.1:
				self.error = True
				self.error_cx = points[0][1] + (points[0][3] // 2)
				self.error_cy = points[0][2] + (points[0][4] // 2) + y_min - 30
		else: self.error_time = 0

		if max_rect is not None:
			(x_min, y_min), (w_min, h_min), angle = max_rect

			angle = self.ang_normilize(w_min, h_min, angle)
			error = x_min - (ww / 2)
			#print(round(angle, 2), round(error, 2))
			#if angle > 15.0:
			#	vx_speed = 0.0
			#else: vx_speed = 0.1
			vx_speed = 0.1
			self.flight.set_velocity(vx=0.05, vy=error*(-0.006), vz=0.0, yaw=float('nan'), yaw_rate=angle*(-0.006), frame_id='body')
			#t = self.flight.telemetry(frame_id = 'aruco_map')
			#self.flight.set_position(x=vx_speed, y=error*(-0.015), z=0.0, yaw=float('nan'), yaw_rate=angle*(-0.015), frame_id='body')
			if self.line_end_time != 0.0 and not self.line_end: 
				self.line_end_time = 0.0
		else:
			if self.line_end_time == 0.0 and not self.line_end:
				self.line_end_time = time.time()
				self.line_end = False
			elif self.line_end_time > 0.0 and (time.time() - self.line_end_time) >= 2.0:
				self.line_end = True
		return image_draw, defect_image

	# детект разливов
	def oil_detect(self, image):
		oil_image = image.copy()
		h, w, _ = image.shape
		blur = cv2.GaussianBlur(image,(5,5), 0)
		hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
		bin_line = cv2.inRange(hsv, (23, 48, 141), (52, 255, 255))

		bin_colors = cv2.inRange(hsv, (0, 83, 13), (255, 255, 255))
		bin = cv2.bitwise_xor(bin_colors, bin_line)

		kernel = np.ones((5,5),np.uint8)
		bin = cv2.erode(bin, kernel)

		y_st = max(0, self.error_cy - 90)
		y_ed = min(h, self.error_cy + 90)
		#image = cv2.circle(image, (self.error_cx, self.error_cy), 5, (127, 0, 255), -1)
		bin = bin[y_st:y_ed, :]

		kernel = np.ones((3,3),np.uint8)
		bin = cv2.erode(bin, kernel)
		bin = cv2.dilate(bin, kernel)

		#cv2.imshow('bin1', bin)
		#cv2.waitKey(1)

		contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		coords = []
		for cnt in contours:
			area = cv2.contourArea(cnt)
			#print(area)
			if cv2.contourArea(cnt) > 200:
				x, y, w, h = cv2.boundingRect(cnt)

				coords.append([sqrt((self.error_cx - x) ** 2 + (self.error_cy - y) ** 2), cnt, area])
				#image[self.error_cy - 90:self.error_cy + 90, :] = cv2.drawContours(image[self.error_cy - 90:self.error_cy + 90, :], cnt,-1,(127,0,255),3)
		coords = sorted(coords, key = lambda x: x[0])
		if len(coords) > 0:
			if self.oil_time == 0: 
				self.oil_time = time.time()
			elif (time.time() - self.oil_time) >= 0.1:
				self.oil = True
				self.oil_area = coords[0][2]
			oil_image[self.error_cy - 90:self.error_cy + 90, :] = cv2.drawContours(oil_image[self.error_cy - 90:self.error_cy + 90, :],[coords[0][1]] ,0,(255, 0, 0),3)
			#image[self.error_cy - 90:self.error_cy + 90, :] = cv2.drawContours(image[self.error_cy - 90:self.error_cy + 90, :],[coords[0][1]] ,0,(0,0,255),2)
		else: self.oil_time = 0
		return oil_image

# фукнции для вывода инфы
def oil_area(area):
	print(f'oil area: {area}')
def defect(x, y):
	print(f'defect: {x} {y}')


rospy.init_node('flight_node')

# сегодня не будем записывать видео, так что указываем 
# record = False
task = Task(record = False, output_file = 'new.mp4')

height = 0.5
height_low_detect = 1.0

# взлетаем на высоту 0.5 метра над зоной старта 
task.flight.navigate_block(z = height, frame_id = 'body', speed = 0.5, tolerance = 0.23, auto_arm=True)
rospy.sleep(1)

# начинаем детектить QR-код
task.QR_detect = True
rospy.sleep(6)

# повторяем попытку на более низкой высоте, если не удалось детектировать QR-код
if len(task.nav_area) == 0:
	task.flight.navigate_block(z = -0.15, frame_id = 'body', speed = 0.3, tolerance = 0.18)
	rospy.sleep(5)

	# если опять не удалось, то проходим небольшое расстояние по диагонали и повторяем попытку 
	if len(task.nav_area) == 0:
		task.flight.navigate_block(x = 0.3, y= 0.3, z = 0, frame_id = 'body', speed = 0.3, tolerance = 0.18)
		rospy.sleep(5)

		# и садим коптер, если опять потерпели неудачу 
		if len(task.nav_area) == 0:
			task.flight.land()
			exit()
		#print('Start moving by square...')
		#t = task.flight.telemetry(frame_id = 'aruco_map')
		#width_x = 1.0
		#width_y = 1.0
		#square = [(t.x + width_x, t.y)]
		#while True:
		#	if len(task.nav_area) > 0:
		#		break
# выключаем детект QR-кодов
task.QR_detect = False

#task.nav_area = [2.5, 0.5]
#task.nav_area = [2.27, 2.29]

#task.flight.navigate_block(z = 1.0, frame_id = 'body', speed = 0.3, tolerance = 0.15)

# выводим содержимое QR-кода
print(f'Qr-code info: x={task.nav_area[0]} y={task.nav_area[1]}')

# запоминаем координаты старта
start_pose = task.flight.telemetry(frame_id = 'aruco_map')
print(f'Start pose by aruco x={start_pose.x} y={start_pose.y}')

# летим к началу линии
error_t = task.flight.telemetry(frame_id = 'aruco_map')
task.flight.navigate_block(x = float(task.nav_area[0]), y = float(task.nav_area[1]), z = height_low_detect, tolerance = 0.18, speed = 0.2, frame_id = 'aruco_map')

rospy.sleep(2.0)

# включаем детект линии и повреждений
answer = []
task.Line_detect = True

# детектим линию, пока она не закончится
while not task.line_end:
	t = task.flight.telemetry(frame_id = 'aruco_map')

	# если нашли повреждение и прошли больше 1 метра от предыдущей позиции,
	# то начинаем детект разлива
	if task.error and sqrt((t.x - error_t.x) ** 2 + (t.y - error_t.y) ** 2) >= 0.8:
		error_t = t

		# выводим информацию о повреждении
		defect(round(t.x, 2), round(t.y, 2))
		
		# начинаем детект разлива
		task.Oil_detect = True
		rospy.sleep(1)
		task.Oil_detect = False

		# формируем массив для финального вывода
		area = 0
		if task.oil:
			oil_area(task.oil_area)
			area = task.oil_area
		answer.append([(t.x, t.y), area])
		task.error = False
		task.oil = False
	rospy.sleep(0.1)

# летим на старт
task.flight.navigate_block(x = float(start_pose.x), y = float(start_pose.y), z = 1.5, tolerance = 0.18, speed = 0.2, frame_id = 'aruco_map')
#task.Land_detect = True
#while task.Land_detect:
#	rospy.sleep(0.1)

# садим коптер (он не сядет)
rospy.sleep(1.0)
task.flight.land()

# выводим финальное сообщение
print(f'Navigation area x={task.nav_area[0]}, y={task.nav_area[1]}')
for i in range(0, len(answer)):
	print(f'{i + 1}. x={answer[i][0][0]}, y={answer[i][0][1]}, S={answer[i][2]}')