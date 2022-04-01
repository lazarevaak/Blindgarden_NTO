import os
import time

from math import sqrt, radians, fmod, pi, isnan
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
		print(f'Video file name {name}')
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
					speed = 0.3, auto_arm = False, frame_id = 'body', tolerance = 0.2):
		if tolerance <= 0.05:
			print(f'Do not indicate small tolerance! (tolarnce={tolerance})')
		print(f'Moving to point x={x} y={y} z={z} ...', end = '', flush=True)
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

		print('Copter is landing...', end = '', flush=True)
		if not res.success: 
			print('Error!')
			return res

		while self.telemetry().armed:
			rospy.sleep(0.2)
		print('Done!')
		return res

# Основной класс, реализует распознавание всего, что нужно
class Task:
	def __init__(self, record = False, output_file = ''):
		# Для управления коптером
		self.flight = Flight()

		# если необходимо записать видео, то создаем экземпляр класса
		self.record = record
		if self.record:
			self.video = VideoSaver()
			self.video.create(output = output_file)

		# флаги для включения распознавания отдельных компонентов
		self.QR_detect = False
		self.Line_detect = False
		self.Oil_detect = False
		self.Land_start = False

		self.height_aruco = 1.0
		self.is_reversing = False
		self.reverse_yaw = 0.0
		self.count_reverse = 0

		# True - right
		# False - left
		self.line_choose = True

		# координаты начала линии с QR-кода
		self.nav_area = []
		self.lake_area = []

		# время, прошедшее с начала первого детектирования повреждения
		self.error_time = 0
		# флаг для отображения статуса детектирования повреждений
		self.error = False
		# максимальный и минимальный порог площади повреждений
		self.error_area = (10, 750)
		# время, которое должно пройти с нахождения повреждения для его учета
		self.error_thr = 0.1

		# флаг указывающий кончилась ли линия
		self.line_end = False
		# время, прошедшее с начала исчезновения линии
		self.line_end_time = 0
		# коэффициент поворота за линией
		self.k_angle = -0.006
		# коэффициент движения по оси Y за линией
		self.k_velocity_y = -0.006
		# скорость движения за линией
		self.line_velocity = 0.097
		# время, которое должно пройти с исчезновения линии для возвращения на точку старта
		self.line_end_thr = 15.0

		self.first_reverse = True

		# координаты поврждения на изображении
		self.error_cx = 0
		self.error_cy = 0

		# координаты центра посадочной площадки
		self.cx = 0
		self.cy = 0

		self.cross_cx = float('nan')
		self.cross_cy = float('nan')
		self.cross_yaw = float('nan')
		self.cross_road = False

		# коэффициент для ограничения скорости выравнивания перед посадкой
		self.land_k = 15.0

		# время, прошедшее с начала детектирования разлива
		self.oil_time = 0
		# флаг, указывающий наличие разлива
		self.oil = False
		# площадь разлива в пикселях
		self.oil_area = 0.0
		# время, которое должно пройти с нахождения разлива для его учета
		self.oil_thr = 0.1

		# цвета для обводки распознанных объектов
		self.qr_color = (0, 255, 0)
		self.vector_line = (127, 127, 0)
		self.line_color = (0, 127, 255)
		self.defect_color = (180, 105, 255)
		self.oil_color = (255, 0, 0)
		self.land_color = (0, 0, 255)
		self.line_width = 3

		# для вывода кадров
		# топик для тестирования работы алгоритма распознавания разметки,
		# повреждений, разливов и QR-кодов
		self.cv_debug = rospy.Publisher('/CV_debug', Image, queue_size=1)
		# необходим по условию задачи, только для отображения контура повреждений
		self.defect_topic = rospy.Publisher('/defect_detect', Image, queue_size=1)
		# также нужен по условию, только для отображения контура разливов
		self.oil_topic = rospy.Publisher('/oil_detect', Image, queue_size=1)

		# для приема кадров с камера
		# используем топик image_raw_throttled
		# для лучшей скорости работа программы, чтобы не нагружать raspberry
		# частота 5Гц
		self.bridge = CvBridge()
		# _throttled
		rospy.Subscriber('main_camera/image_raw_throttled', \
				Image, self.image_cb)

	# функция нормализующая координаты от 0 до 1
	def normilize(self, x, y):
		dst = sqrt(x ** 2 + y ** 2)
		return x / dst, y / dst

	# callback функция для приема кадров
	def image_cb(self, msg):
		# чтение кадра
		try:
			image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
		except CvBridgeError as e: print(e)

		# если записываем видео
		if self.record: self.video.write(image)

		# для тестирования алгоритмов распознавания
		cv_debug = image.copy()
		defect_image = image.copy()
		oil_image = image.copy()


		# если распознаем QR-код
		if self.QR_detect:
			cv_debug = self.qr_code(image)

		# если распознаем разметку и повреждения
		if self.Line_detect:
			cv_debug, defect_image = self.line_detect(image)

		# если распознаем QR-код
		if self.Oil_detect:
			cv_debug, oil_image = self.oil_detect(image, cv_debug)

		# если распознаем посадочное место
		if self.Land_start:
			cv_debug = self.land_detect(image)

		# выводим изображения в топики
		try:
			self.cv_debug.publish(self.bridge.cv2_to_imgmsg(cv_debug, encoding = 'bgr8'))
			self.defect_topic.publish(self.bridge.cv2_to_imgmsg(defect_image, encoding = 'bgr8'))
			self.oil_topic.publish(self.bridge.cv2_to_imgmsg(oil_image, encoding = 'bgr8'))
		except CvBridgeError as e: print(e)

	# распознавание QR-кодов (pyzbar)
	def qr_code(self, image):
		boxes = pyzbar.decode(image)
		if len(boxes) == 0:
			return image
		
		# обрабатываем информацию из QR-кодов
		for box in boxes:
			data = box.data.decode("utf-8").split('\n')
			if len(data) != 2:
				print('Error data format in Qr-code, skipping!')
				return image

			image = cv2.rectangle(image, (box.rect.left, box.rect.top), \
					(box.rect.width + box.rect.left, box.rect.height + box.rect.top), \
					self.qr_color, self.line_width)
			
			#if len(self.nav_area) != 0:
			#	return image

			self.nav_area = data[0].split(' ')
			self.lake_area = data[1].split(' ')
			# нашли QR-код, отключили распознавание
			self.QR_detect = False
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

	def ang_norm(self, ang):
		a = fmod(fmod(ang, 2.0 * pi) + 2.0 * pi, 2.0 * pi)
		if a > pi:
			a -= 2.0 * pi
		return a

	# распознавание линии и повреждений
	def line_detect(self, image):
		# для рисования разметки и повреждений
		image_draw = image.copy()
		defect_image = image.copy()

		if self.line_end:
			return image_draw, defect_image

		if self.is_reversing:
			pose = self.flight.telemetry(frame_id = 'aruco_map')
			ang_min = self.ang_norm(self.reverse_yaw - 0.15)
			ang_max = self.ang_norm(self.reverse_yaw + 0.15)

			if pose.yaw >= ang_min and pose.yaw <= ang_max:
				self.is_reversing = False
				self.first_reverse = False

			else:
				return image_draw, defect_image

		if self.cross_road:
			pose = self.flight.telemetry(frame_id = 'navigate_target')
			if sqrt(pose.x ** 2 + pose.y ** 2 + pose.z ** 2) <= 0.2:
				self.cross_road = False
			else:
				return image_draw, defect_image

		height, width, _ = image.shape

		blur = cv2.GaussianBlur(image, (5,5), 0)

		# бинаризуем изображение из пространства HSV
		# в этом пространстве легче выделить желтый цвет
		hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
		bin = cv2.inRange(hsv, \
			(23, 48, 141), (52, 180, 255))

		bin = bin[(height // 2) - 60:(height // 2) + 90, :]
		kernel = np.ones((5,5),np.uint8)
		bin = cv2.erode(bin, kernel)
		bin = cv2.dilate(bin, kernel)

		# ищем контуры линии
		contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		coords_y = []
		compute_rect = [float('inf'), (0, 0), (0, 0), 0, (0, 0), (0, 0)]
		for cnt in contours:
			# фильтрация по площади в пикселях
			area = cv2.contourArea(cnt)
			if cv2.contourArea(cnt) > 300:
				rect = cv2.minAreaRect(cnt)
				bx, by, bw, bh = cv2.boundingRect(cnt)
				(x_min, y_min), (w_min, h_min), angle = rect

				# фильтрация по соотношению сторон
				ratio = max(h_min / w_min, w_min / h_min)
				if True:
				#if abs(1.0 - ratio) > 0.1:
					# извлекаем координаты точек прямоугольника для 
					# дальнейшего определения повреждений
					box = cv2.boxPoints(rect)
					box = np.int0(box)
					# смещаем точки для корректного отображения
					box = [[p[0], p[1] + (height // 2) - 60] for p in box]

					# рисуем
					box = np.array(box)
					image_draw = cv2.drawContours(image_draw, [box], 0, self.line_color, self.line_width)
					image_draw = cv2.circle(image_draw, (int(x_min), int(y_min + (height // 2) - 60)), 5, self.line_color, -1)

					# ищем точки с минимальной и максимальной y координатой
					# для последующего определения повреждений
					box = sorted(box, key = lambda x: x[1])
					coords_y.append([box[0], box[-1]])

					# сохраняем контур с максимальной координатой y для последующего расчета скорости клевера
					if compute_rect[0] > box[0][1]:
						compute_rect = [box[0][1], (x_min, y_min), (w_min, h_min), angle, (bx, by), (bw, bh)]
		
		# если нашли несколько контуров линии, то пытаемся найти повреждение
		if len(coords_y) > 1:
			# находим самый нижний и верхний контур линии
			coords_y = sorted(coords_y, key = lambda x: x[0][1])
			x_min, y_min = int(coords_y[0][1][0]), int(coords_y[0][1][1])
			y_max = int(coords_y[-1][0][1])

			# вырезаем участок изображения, на котором ищем повреждение
			# рассчитываем исходя из y координат контуров линии
			spl_start = max(0, y_min - 30)
			spl_end = min(y_max + 30, height)
			mask_search = image[spl_start:spl_end, :, 1].copy()
			mask_search = cv2.inRange(mask_search, 0, 130)
			kernel = np.ones((3,3),np.uint8)
			mask_search = cv2.erode(mask_search, kernel)

			# ищем контуры повреждений
			contours, _ = cv2.findContours(mask_search, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			
			error_obj = [float('inf'), (0, 0), (0, 0), 0]
			y_min_local = y_min - (y_min - 30)
			for cnt in contours:
				# фильтр по площади
				if cv2.contourArea(cnt) > self.error_area[0] and cv2.contourArea(cnt) < self.error_area[1]:
					# находим мин. обрамляющюю контура
					x, y, w, h = cv2.boundingRect(cnt)

					dst = sqrt((30 - y) ** 2 + (x_min - x) ** 2)
					if error_obj[0] > dst:
						error_obj = [dst, (x + (w // 2), y + (h // 2) + y_min - 30), cnt]

			if error_obj[0] != float('inf'):
				# рисуем контур повреждения
				defect_image[y_min - 30:y_max + 30, :] = cv2.drawContours(defect_image[y_min - 30:y_max + 30, :], \
						[error_obj[2]], 0, self.defect_color, self.line_width)
				image_draw[y_min - 30:y_max + 30, :] = cv2.drawContours(image_draw[y_min - 30:y_max + 30, :], \
						[error_obj[2]], 0, self.defect_color, self.line_width)
			
				now = time.time()
				if self.error_time == 0.0: self.error_time = now
				elif (now - self.error_time) >= self.error_thr:
					self.error = True
					self.error_cx, self.error_cy = error_obj[1]
			else: self.error_time = 0.0
		else: self.error_time = 0.0

		# рассчитываем скорости коптера для движения за линией
		if compute_rect[0] != float('inf'):
			_, (x_min, y_min), (w_min, h_min), angle, (bx, by), (bw, bh) = compute_rect
			
			# если линия перевернута на 180 градусов, то поворачиваемся
			angle = self.ang_normilize(w_min, h_min, angle)
			y_min += (height // 2) - 60

			#  and self.first_reverse
			frame_cn = (height / 2)
			thr = frame_cn + (frame_cn / 3)
			thr_low = frame_cn - 15
			if (self.count_reverse == 0 and y_min >= thr_low) or (self.count_reverse >= 1 and y_min >= thr):
				self.count_reverse += 1
				if self.count_reverse >= 2:
					if self.count_reverse == 2 and (not isnan(self.cross_cx)):
						print('returning to crossroad')
						pose = self.flight.telemetry(frame_id = 'aruco_map')
						self.flight.navigate(x = self.cross_cx, y = self.cross_cy, z = pose.z, \
							speed = 0.2, yaw = self.cross_yaw, yaw_rate = 0.0, frame_id='aruco_map')

						self.cross_road = True
						self.line_choose = False

					if self.count_reverse == 3 or isnan(self.cross_cx):
						print('end line')
						self.line_end = True

						self.flight.set_velocity(vx = 0.0, vy = 0.0, vz = 0.0, \
							yaw = float('nan'), yaw_rate = 0.0, frame_id = 'body')

				else:
					pose = self.flight.telemetry(frame_id = 'aruco_map')
					need_yaw = self.ang_norm(pose.yaw + pi)

					print('reverse line')
					#self.flight.navigate_block(yaw = radians(180.0), frame_id = 'body', speed = 0.2, tolerance = 0.2)
					#angle_vel = 10.0 * self.k_angle
					#self.flight.set_velocity(vx = 0.0, vy = 0.0, vz = 0.0, \
					#	yaw = float('nan'), yaw_rate = angle_vel, frame_id = 'body')
					#self.flight.set_position(x = 0.0, y = 0.0, z = 0.0, \
					#	yaw = float('nan'), yaw_rate = angle_vel, frame_id = 'body')

					self.flight.navigate(x = pose.x, y = pose.y, z = pose.z, \
						yaw = need_yaw, frame_id='aruco_map')
					self.reverse_yaw = need_yaw
					self.is_reversing = True
			else:
				if self.count_reverse == 0:
					self.count_reverse += 1

				y_it = by + (bh // 4) 
				x_it = bx
				center = 0
				if (not self.line_choose):
					center = width
				center_cnt = 0
				while x_it < (bw + bx):
					if bin[y_it][x_it] == 255:
						start_point = x_it
						nBlackPoints = 3
						while nBlackPoints > 1 and (x_it + 3) < (bx + bw):
							x_it += 3
							nBlackPoints = 0
							for j in range(0, 3):
								nBlackPoints += int(bin[y_it][x_it - j] == 255)
						#print((x_it - start_point))
						if (x_it - start_point) >= 5:
							center_cnt += 1
							cn = (x_it + start_point) / 2
							if self.line_choose and cn > center:
								center = cn
							elif (not self.line_choose) and cn < center:
								center = cn
					x_it += 1

				if center_cnt >= 2 and isnan(self.cross_cx):
					pose = self.flight.telemetry(frame_id = 'aruco_map')
					self.cross_cx = pose.x
					self.cross_cy = pose.y
					self.cross_yaw = pose.yaw
					print(f'Cross road x={round(self.cross_cx, 2)}, y={round(self.cross_cy, 2)}, yaw={round(self.cross_yaw, 4)}')


				image_draw = cv2.circle(image_draw, (int(center), int(y_it + (height // 2) - 60)), 8, (127, 127, 127), -1)
				#spl_start = int(max(bx, center - 40))
				#spl_end = int(min(bx + bw, center + 40))
				#mask_ln = bin[by:(y_it), spl_start:spl_end]

				#contours, _ = cv2.findContours(mask_ln, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				#contours = sorted(contours, key = cv2.contourArea)
				#if len(contours) > 0:
				#	cnt = contours[-1]
				#	_, (w_mn, h_mn), angle = cv2.minAreaRect(cnt)
				#	angle = self.ang_normilize(w_mn, h_mn, angle)


				#	image_draw[(height // 2) - 60:(height // 2) + 90, :][by:(y_it), spl_start:spl_end] = \
				#		cv2.drawContours(image_draw[(height // 2) - 60:(height // 2) + 90, :][by:(y_it), spl_start:spl_end], \
				#			[cnt], 0, (255, 0, 255), self.line_width)


				self.first_reverse = False
				#print('angle', angle)
				#error = x_min - (width / 2)
				error = center - (width / 2)

				self.flight.set_velocity(vx = self.line_velocity, vy = error * self.k_velocity_y, vz = 0.0, \
					yaw = float('nan'), yaw_rate = angle * self.k_angle, frame_id = 'body')
			
			self.line_end_time = int(self.line_end) * time.time()
		else:
			now = time.time()
			if self.line_end_time == 0 and (not self.line_end): self.line_end_time = now
			elif self.line_end_time > 0.0 and (now - self.line_end_time) >= self.line_end_thr:
				self.line_end = True
		return image_draw, defect_image

	# детект разливов
	def oil_detect(self, image, draw):
		# для рисования
		draw = draw.copy()
		oil_image = image.copy()
		height, width, _ = image.shape

		# бинаризуем из цветового пространства HSV по пороговым значениям
		blur = cv2.GaussianBlur(image,(5,5), 0)
		hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
		bin_line = cv2.inRange(hsv, (23, 48, 141), (52, 255, 255))
		bin_colors = cv2.inRange(hsv, (0, 48, 141), (52, 255, 255))

		# применяем операцию 'исключающего или' для того, чтобы убрать линию
		bin = cv2.bitwise_xor(bin_colors, bin_line)
		kernel = np.ones((5,5),np.uint8)
		bin = cv2.erode(bin, kernel)

		# вырезаем участок изображения, на котором введем поиск разлива
		y_st = max(0, self.error_cy - 90)
		y_ed = min(height, self.error_cy + 90)
		bin = bin[y_st:y_ed, :]
		kernel = np.ones((3,3),np.uint8)
		bin = cv2.erode(bin, kernel)
		bin = cv2.dilate(bin, kernel)

		# поиск контуров разливов
		contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		obj = [float('inf'), 0, 0]
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if cv2.contourArea(cnt) > 750:
				x, y, w, h = cv2.boundingRect(cnt)

				dst = sqrt((self.error_cx - x) ** 2 + (self.error_cy - y) ** 2)
				if obj[0] >  dst:
					obj = [dst, cnt, area]

		if obj[0] != float('inf'):
			now = time.time()
			if self.oil_time == 0 and (not self.oil): self.oil_time = now
			elif self.oil_time > 0 and (not self.oil) and (now - self.oil_time) >= self.oil_thr:
				self.oil = True
				self.oil_area = obj[2]
				self.oil_time = 0

			oil_image[self.error_cy - 90:self.error_cy + 90, :] = \
				cv2.drawContours(oil_image[self.error_cy - 90:self.error_cy + 90, :],\
					[obj[1]], 0, self.oil_color, self.line_width)
			draw[self.error_cy - 90:self.error_cy + 90, :] = \
				cv2.drawContours(draw[self.error_cy - 90:self.error_cy + 90, :],\
					[obj[1]], 0, self.oil_color, self.line_width)
		else: self.oil_time = 0

		return draw, oil_image

	# детектирует здание для посадки
	def land_detect(self, image):
		# для рисования
		image_draw = image.copy()
		height, width, _ = image.shape

		# проводим бинаризацию по порогу
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		bin = cv2.inRange(gray, 200, 255)

		# обрезаем изображение по бокам, чтобы убрать лишнии объекты
		corner = 30
		kernel = np.ones((1, 1), 'uint8')
		bin = bin[corner:(height - corner), corner:(width - corner)]
		bin = cv2.erode(bin, kernel)
		bin = cv2.dilate(bin, kernel)

		# ищем контур здания
		contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		obj = [-1, (0, 0), (0, 0)]
		for cnt in contours:
			# аппроксимируем контур для дальнейшей работы
			epsilon = 0.1 * cv2.arcLength(cnt, True)
			approximations = cv2.approxPolyDP(cnt, epsilon, True)
			area = cv2.contourArea(approximations)
			
			# фильтрация по площади
			if area > 1500:
				x, y, w, h = cv2.boundingRect(approximations)
				
				# фильтрация по соотношению сторон и выпуклости контура
				ratio = max(w / h, h / w)
				if abs(1.0 - ratio) > 0.1 and cv2.isContourConvex(approximations):
					if obj[0] < area:
						obj = [area, (x + corner, y + corner), (x + corner + w, y + corner + h)]

		if obj[0] != -1:
			self.cx = (obj[1][0] + obj[2][0]) / 2
			self.cy = (obj[1][1] + obj[2][1]) / 2
			#print(obj[1][0], obj[1][1], obj[2][0], obj[2][1])
			image_draw = cv2.rectangle(image_draw, (obj[1][0], obj[1][1]), (obj[2][0], obj[2][1]), \
				self.land_color, self.line_width)

			x, y = self.cx - (width / 2), self.cy - (height / 2)
			if sqrt(x ** 2 + y ** 2) <= 15:
				self.Land_start = False
				self.flight.set_velocity(vx = 0.0, vy = 0.0, vz = 0.0, \
					yaw = float('nan'), yaw_rate = 0.0, frame_id = 'body')
			else:
				x, y = self.normilize(x, y)
				x /= -self.land_k
				y /= -self.land_k
				self.flight.set_velocity(vx = y, vy = x, vz = 0.0, \
					yaw = float('nan'), yaw_rate = 0.0, frame_id = 'body')

		return image_draw

# фукнции для вывода инфы
def oil_area(area):
	print(f'oil area: {area}')
def defect(x, y):
	print(f'defect: {x} {y}')
def lake_suc():
	print('Successful water withdrawal')


rospy.init_node('flight_node')

# флаги для тестирования отдельных элементов кода
TAKEOFF = True
QR_CODE = True
QR_CODE_FAKE = [2.5, 0.5]
QR_CODE_FAKE_LAKE = [1.0, 1.0]
LAKE_TASK = True
LINE_DETECT = True
LAND = True

# высота взлета
height_start = 0.45
#height_start = 1.3
# высота движения на поле
height_line = 1.2
height_aruco = 1.5
height_lake = 0.5
# высота при посадке
height_landing = 1.5

# сегодня не будем записывать видео, так что указываем 
# record = False
task = Task(record = False, output_file = 'newday5.mp4')
task.height_aruco = height_aruco

if TAKEOFF:
	# взлетаем на высоту 0.5 метра над зоной старта 
	task.flight.navigate_block(z = height_start, frame_id = 'body', speed = 0.5, tolerance = 0.2, auto_arm=True)
	#task.flight.navigate_block(yaw = radians(180.0), frame_id = 'body', speed = 0.5, tolerance = 0.2)
	rospy.sleep(1)

if QR_CODE:

	# начинаем детектить QR-код
	task.QR_detect = True
	ts = rospy.Time.now()
	while len(task.nav_area) == 0:
		if (rospy.Time.now() - ts) > rospy.Duration(secs = 10.0):
			break
		rospy.sleep(0.2)

	# если не нашли qr-код, то двигаемся по квадрату и снижаем высоту
	if len(task.nav_area) == 0:
		print('QR-code not find, start moving by square')
		#task.flight.navigate_block(z = -0.15, frame_id = 'body', speed = 0.3, tolerance = 0.2)
		#rospy.sleep(1)

		pose = task.flight.telemetry(frame_id = 'aruco_map')
		square_size = 1.0
		square = [[pose.x + (square_size / 2), pose.y], 
				[pose.x + (square_size / 2), pose.y + (square_size / 2)], 
				[pose.x - (square_size / 2), pose.y + (square_size / 2)], 
				[pose.x - (square_size / 2), pose.y - (square_size / 2)], 
				[pose.x + (square_size / 2), pose.y - (square_size / 2)]]

		for i in range(0, len(square)):
			square[i][0] = round(max(0.0, square[i][0]), 2)
			square[i][0] = round(min(3.0, square[i][0]), 2)

			square[i][1] = round(max(0.0, square[i][1]), 2)
			square[i][1] = round(min(4.0, square[i][1]), 2)

		for i in range(5):
			for x, y in square:
				task.flight.navigate_block(x = x, y= y, z = 1.5, frame_id = 'aruco_map', speed = 0.3, tolerance = 0.2)
				
				ts = rospy.Time.now()
				while len(task.nav_area) == 0:
					if (rospy.Time.now() - ts) > rospy.Duration(secs = 5.0):
						break
					rospy.sleep(0.2)

				if len(task.nav_area) != 0: break

			if len(task.nav_area) != 0: break

	if len(task.nav_area) == 0:
		print(f'QR-code not find!')
		task.flight.land()
		exit()
	# выключаем детект QR-кодов
	task.QR_detect = False

	# выводим содержимое QR-кода
elif len(QR_CODE_FAKE) == 2:
	task.nav_area = QR_CODE_FAKE
	task.lake_area = QR_CODE_FAKE_LAKE

print(f'Navigation area: x={task.nav_area[0]}, y={task.nav_area[1]}')
print(f'Lake center x={task.lake_area[0]}, y={task.lake_area[1]}')

# запоминаем координаты старта
start_pose = task.flight.telemetry(frame_id = 'aruco_map')
print(f'Start pose by aruco x={start_pose.x}, y={start_pose.y}')

if LINE_DETECT:
	print('Going to line start')
	# летим к началу линии
	#error_t = task.flight.telemetry(frame_id = 'aruco_map')
	#task.flight.navigate_block(x = float(task.nav_area[0]), y = float(task.nav_area[1]), z = height_aruco, tolerance = 0.18, speed = 0.2, frame_id = 'aruco_map')
	#rospy.sleep(1)
	task.flight.navigate_block(x = float(task.nav_area[0]), y = float(task.nav_area[1]), z = height_line, tolerance = 0.18, speed = 0.2, frame_id = 'aruco_map')
	rospy.sleep(1)

	AREA_CONST = 26818.18

	# включаем детект линии и повреждений
	answer = []
	task.Line_detect = True
	task.Oil_detect = True
	# детектим линию, пока она не закончится
	while not task.line_end:
		if (not task.is_reversing) and (not task.cross_road):
			try:
				t = task.flight.telemetry(frame_id = 'aruco_map')
			except:
				continue

			# если нашли повреждение и прошли больше 1 метра от предыдущей позиции,
			# то начинаем детект разлива
			if task.error:
				dst = 0.0
				if len(answer) > 0:
					dst = sqrt((t.x - answer[-1][2].x) ** 2 + \
						(t.y - answer[-1][2].y) ** 2)
				if len(answer) == 0 or dst >= 0.6:

					# выводим информацию о повреждении
					defect(round(t.x, 2), round(t.y, 2))
					
					# начинаем детект разлива
					task.Oil_detect = True
					rospy.sleep(4)
					task.Oil_detect = False

					# формируем массив для финального вывода
					area = 0
					if task.oil:
						oil_area(task.oil_area)
						area = task.oil_area
					answer.append([(t.x, t.y), round(area / AREA_CONST, 5), t])
					task.error = False
					task.oil = False
					task.oil_area = 0.0
		rospy.sleep(0.1)

	task.Line_detect = False
	task.Oil_detect = False

if LAKE_TASK:
	print('Going to lake center...')
	task.flight.navigate_block(x = float(task.lake_area[0]), y = float(task.lake_area[1]), z = height_aruco, tolerance = 0.18, speed = 0.2, frame_id = 'aruco_map')
	rospy.sleep(2)
	task.flight.navigate_block(z = (height_lake - height_aruco), tolerance = 0.2, speed = 0.2, frame_id = 'body')
	rospy.sleep(5)
	lake_suc()

	task.flight.navigate_block(z = (height_aruco - height_lake), tolerance = 0.2, speed = 0.2, frame_id = 'body')
	rospy.sleep(2)

if LAND:
	# летим на старт
	task.flight.navigate_block(x = float(start_pose.x), y = float(start_pose.y), z = height_landing, \
		tolerance = 0.2, speed = 0.2, frame_id = 'aruco_map')
	time.sleep(1.45)
	print('Start landing...')

	#task.Land_start = True
	#while task.Land_start:
	#	rospy.sleep(0.2)
	#task.flight.disarm()
	task.flight.land()

if LINE_DETECT:

	# выводим финальное сообщение
	print(f'Navigation area x={task.nav_area[0]}, y={task.nav_area[1]}')
	print(f'Lake center x={task.lake_area[0]}, y={task.lake_area[1]}')
	for i in range(0, len(answer)):
		print(f'{i + 1}. x={answer[i][0][0]}, y={answer[i][0][1]}, S={answer[i][1]}')
