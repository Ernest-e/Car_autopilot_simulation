#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 19:47:44 2024

@author: amina
"""

import pygame
import math
import numpy as np
import random
import torch
import torch.nn as nn
from deap import base, algorithms
from deap import creator
from deap import tools
from deap.algorithms import varAnd
import matplotlib.pyplot as plt
import time

# Class for track
class Track:
    def __init__(self, points, widths):
        self.points = points
        self.widths = widths


# Class for car
class Car:
    def __init__(self, x, y, angle=0, speed = 0, acceleration = 0, width=10, length=10, max_speed=2):
        self.init_x = x
        self.init_y = y
        self.init_angle = angle
        self.init_speed = speed
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.acceleration = acceleration
        self.width = width
        self.length = length
        self.position = (self.x, self.y)
        self.max_speed = max_speed

    def update(self):
        self.speed = max(0, min(self.speed, self.max_speed))
        
        self.x += math.cos(math.radians(self.angle%360)) * self.speed
        self.y += math.sin(math.radians(self.angle%360)) * self.speed
        

    def reset(self, x, y):
        """
        Resetting the vehicle status to the default
        """
        self.init_x = x
        self.init_y = y
        self.x = self.init_x
        self.y = self.init_y
        self.angle = self.init_angle
        self.speed = self.init_speed
        
    
    # the rendering function
    def draw(self, screen):
        pygame.draw.polygon(screen, RED, [(self.x + math.cos(math.radians(self.angle)) * self.length, self.y + math.sin(math.radians(self.angle)) * self.length),
                                            (self.x + math.cos(math.radians(self.angle + 120)) * self.width, self.y + math.sin(math.radians(self.angle + 120)) * self.width),
                                            (self.x + math.cos(math.radians(self.angle + 240)) * self.width, self.y + math.sin(math.radians(self.angle + 240)) * self.width)])
def find_intersection(ray_origin, ray_direction, point1, point2):
    #The vector of the ray direction
    v1 = ray_origin - point1
    v2 = point2 - point1
    v3 = np.array([-ray_direction[1], ray_direction[0]])

    dot_product = np.dot(v2, v3)
    if dot_product == 0:
        return None  # The vectors are parallel, there is no intersection

    t1 = np.cross(v2, v1) / dot_product
    t2 = np.dot(v1, v3) / dot_product

    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
        return ray_origin + t1 * ray_direction
    return None

def lidar_scan(car_pos, car_angle, track_points, track_widths, num_rays=4, max_distance=100):
    """
    Simulates lidar by scanning the environment for the presence of track boundaries.
    """
    angles = np.linspace(0, 360, num_rays, endpoint=False) + car_angle
    distances = []

    for angle in angles:
        # The direction of the ray, taking into account the angle of the car
        ray_dir = np.array([math.cos(math.radians(angle%360)), math.sin(math.radians(angle%360))])
        ray_origin = np.array(car_pos)
        closest_intersection = None

        for i in range(len(track_points) - 1):
            point1 = np.array(track_points[i])
            point2 = np.array(track_points[i + 1])

             # find intersections for both sides of the route
            for offset in [-track_widths[i] / 2, track_widths[i] / 2]:
                offset_vec = np.array([point2[1] - point1[1], point1[0] - point2[0]])
                offset_vec = offset * offset_vec / np.linalg.norm(offset_vec)

                intersection = find_intersection(ray_origin, ray_dir, point1 + offset_vec, point2 + offset_vec)
                if intersection is not None:
                    if closest_intersection is None or np.linalg.norm(intersection - ray_origin) < np.linalg.norm(closest_intersection - ray_origin):
                        closest_intersection = intersection

        if closest_intersection is not None:
            distances.append(np.linalg.norm(closest_intersection - ray_origin))
        else:
            distances.append(max_distance)

    return [distance / max_distance for distance in distances] # normalization


def get_car_state(car, track):
    '''
    Getting information about the condition of the car relative to the track
    '''
    max_speed = 2
    car_pos = car.x, car.y
    car_angle = car.angle / 360
    car_speed = car.speed / max_speed
    track_points = track.points
    track_widths = track.widths
    distances = lidar_scan(car_pos, car_angle, track_points, track_widths)
    distances.extend([car_angle, car_speed])

    return torch.tensor(distances, dtype=torch.float32)

def point_to_line_distance(point, line_start, line_end):
    """Calculate the distance from a point to a line segment"""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    den = math.sqrt((y2-y1)**2 + (x2-x1)**2)
    return num / den if den else 0

def is_point_on_segment(point, start, end, buffer=0):
    """
    Check whether the point is on the segment, taking into account a small buffer (the width of the route)
    """
    #Expanding the check for the presence within the rectangle formed by the start and end points of the segment
    px, py = point
    sx, sy = start
    ex, ey = end

    # We take the buffer into account in the calculations
    dx = ex - sx
    dy = ey - sy

    if dx == 0 and dy == 0:   # The segment represents a point
        return math.sqrt((px - sx) ** 2 + (py - sy) ** 2) <= buffer

    # Normalizing the segment
    norm = dx * dx + dy * dy
    u = ((px - sx) * dx + (py - sy) * dy) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # Find the nearest point on the segment to the point
    x = sx + u * dx
    y = sy + u * dy

    dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
    return dist <= buffer


def is_car_off_track(car, track):
    for i in range(len(track.points) - 1):
        if is_point_on_segment((car.x, car.y), track.points[i], track.points[i + 1], buffer=track.widths[i] / 2):
            return False  #  The car is on the track
    return True  # The car is off the track



def neural_network_decision(state, model):
    """
    Make a decision based on the current state of the car and the neural network model.
    
    :param car: car object
    :param model: an instance of a trained neural network model
    :return: speed_change, angle_change
    """
    

    output = model.predict(state)
    
    # Extracting the desired speed and angle changes from the network output
    speed_change, angle_change = output[0], output[1]
    
    return speed_change, angle_change

def line_intersection(line1, line2):
    """
    Defines the intersection point of two lines (if it exists)
    
    :param line1: A tuple of two points (x1, y1), (x2, y2) defining the first line.
    :param line2: A tuple of two points (x3, y3), (x4, y4) defining the second line.
    :return: The coordinates of the intersection point are (x, y) or None if the lines do not intersect.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None  # The lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    if 0 <= t <= 1 and 0 <= u <= 1:
        # The intersection point is located within both segments
        return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

    return None

def has_finished(car_position, prev_car_position, finish_line):
    """
    Check whether the car has crossed the finish line between two consecutive positions.
    
    :param car_position:The current position of the car (x, y).
    :param prev_car_position: The previous position of the machine (x, y).
    :param finish_line: Coordinates of the finish line (x1, y1, x2, y2).
    :return: True, if the car has crossed the finish line; else False.
    """
    intersection = line_intersection((prev_car_position[0], prev_car_position[1], car_position[0], car_position[1]), finish_line)
    return intersection is not None



# pretrained weights
weights = np.array([ 0.97560521,  0.56991716,  0.29677615,  0.76343146,  0.47599494,
        0.57988206,  0.27625482, -0.42671009, -0.79068105, -0.78541129,
        0.56458586,  0.9639349 ,  0.14835754, -0.60670087, -0.65712762,
        0.0764702 ,  0.97172667, -0.95936868, -0.05850595, -0.88406838,
       -0.92032182, -0.08766139,  0.23830389, -0.70035715, -0.97892136,
       -0.7253488 , -0.8773796 , -0.16274916,  0.69432302,  0.12636279,
       -0.80300833,  0.46784555,  0.68489602,  0.40139405,  0.11855042,
       -0.21117145,  0.33301668,  0.25544577,  0.55599297, -0.58689216,
       -0.20833955,  0.24444362,  0.64363421, -0.5823612 , -0.70423551,
        0.25697352, -0.84851762,  0.27690595, -0.06342311, -0.46294389,
        0.06698906, -0.57677885, -0.51310292, -0.04580464, -0.87845016,
        0.20235224, -0.31854818, -0.47030603, -0.85315319,  0.07193979,
        0.70658071,  0.28805047,  0.95076456,  0.2309462 ,  0.67674595,
       -0.61504671, -0.53379261, -0.06972618,  0.20650142,  0.28683442,
       -0.58284622, -0.16572309, -0.82891401, -0.79180269,  0.29728132,
       -0.40392738,  0.66964264,  0.55353672, -0.44753558, -0.45970757,
        0.13929651,  0.67781511,  0.15332251,  0.04851026, -0.39485376,
        0.11035072,  0.56870625,  0.49535772, -0.18476283,  0.97668019,
        0.85973692,  0.80917554,  0.5302339 ,  0.97774741,  0.02571028,
        0.17398089,  0.38702725,  0.104407  ,  0.91995546,  0.32003156,
       -0.95717633, -0.32158125, -0.58465486,  0.03079668, -0.46674942,
       -0.17163782,  0.14580504, -0.13563469, -0.67827729,  0.55400019,
        0.36995066,  0.66994199, -0.83719982, -0.68083454, -0.28224382,
       -0.14603538, -0.8796713 ,  0.32835946, -0.51118197,  0.58922745,
        0.3969415 ,  0.44854346, -0.15748723,  0.27862256, -0.95906763,
        0.85627516, -0.94325922,  0.81377708,  0.11306943, -0.34260863,
       -0.55537371, -0.90434488,  0.78672983, -0.32657025, -0.87630199,
       -0.48120754, -0.66001442,  0.50378575,  0.52462032, -0.38251107,
       -0.13673543,  0.12038835, -0.285126  ,  0.33171664,  0.52322601,
       -0.73231382,  0.10350494,  0.43514408, -0.39961359, -0.40185288,
        0.99286502, -0.94536451, -0.4832937 , -0.43391456, -0.74692625,
       -0.55295173, -0.45844013, -0.42339895, -0.44568591, -0.87263801,
       -0.86573083, -0.69833103,  0.85045792,  0.84202019, -0.10735857,
       -0.57786362,  0.48751975,  0.34438162, -0.64477642, -0.4253078 ,
       -0.09454203,  0.1000054 , -0.18505045, -0.54442282,  0.68321528,
        0.48420471, -0.29994743,  0.84377361, -0.21853136, -0.70313946,
       -0.40943097, -0.63254598, -0.63739001,  0.94132261, -0.45932623,
       -0.85983755,  0.62876429,  0.18358076, -0.28226848,  0.24712935,
        0.11994869, -0.44997573, -0.52682052,  0.85436914, -0.05998068,
       -0.76202779,  0.5614162 , -0.86932813, -0.37229275, -0.97171338,
        0.26409766,  0.84530813,  0.30866684,  0.82700298, -0.70675505,
       -0.61763331, -0.97013037,  0.79118737,  0.68358661,  0.98442053,
       -0.22118171,  0.61675564,  0.38138525, -0.86630517,  0.70845121,
        0.92374837, -0.92804532,  0.47912568, -0.68171496, -0.73634689,
        0.56920325,  0.10302434,  0.11352114,  0.97240482,  0.88038134,
       -0.3042278 , -0.54450991, -0.46250582, -0.152207  ,  0.90227093,
       -0.06903009,  0.89376725, -0.66233767, -0.37231143, -0.75328014,
       -0.76305072,  0.83999247, -0.55391823,  0.1890609 ,  0.16361831,
       -0.53693501,  0.98313516, -0.76619508, -0.76417667, -0.98589715,
        0.26163612, -0.16733448,  0.40330726,  0.99097523,  0.28052762,
        0.53498663, -0.82968484,  0.03747436, -0.05479602,  0.76517119,
       -0.36278136, -0.98846114, -0.57376623, -0.70496641,  0.08904459,
       -0.92829175, -0.02458703,  0.75890292,  0.67891089,  0.58194731,
       -0.86798437,  0.2245186 ,  0.02024709,  0.90513339, -0.94010202,
        0.94695349, -0.40713157,  0.3270267 ,  0.89153866, -0.40052483,
       -0.57073361, -0.80630759,  0.1193834 ,  0.5696142 ,  0.47051506,
       -0.76241685, -0.89913727, -0.01588478, -0.82894256,  0.58204328,
        0.81182351, -0.17784525, -0.83527664,  0.33725756, -0.08690944,
        0.17690218,  0.87413043,  0.71098368, -0.47582555,  0.57795625,
        0.74622581, -0.42130235,  0.21720907, -0.82927589, -0.31930059,
        0.19373097,  0.08099833,  0.34990505,  0.21204892, -0.1571641 ,
       -0.82325261, -0.3986256 ,  0.36253695, -0.65761809,  0.44795046,
       -0.5370487 ,  0.76355373, -0.48425118, -0.79129222,  0.54897756,
       -0.78469443,  0.3996932 , -0.00850184,  0.03180903, -0.18713585,
        0.82074625,  0.44013337, -0.94523205,  0.76349349, -0.18470538,
       -0.85661204, -0.01597604, -0.01082121,  0.60697877,  0.71425372,
       -0.67993294, -0.09724442,  0.91576974, -0.20894149, -0.74412976,
        0.15533024, -0.17855046, -0.52222167, -0.1735383 , -0.6421034 ,
        0.89350345,  0.62418806, -0.92389323, -0.43588975, -0.90941771,
       -0.17401894,  0.5508112 ,  0.7293736 , -0.38531082, -0.69717932,
        0.97939105,  0.53749255,  0.89393417,  0.47623359,  0.21282015,
        0.47436638,  0.82114683, -0.23937444, -0.80047789, -0.95383126,
       -0.39895659,  0.61540185,  0.21247962,  0.91255307, -0.68052551,
       -0.93803711, -0.18031899,  0.42184465, -0.9806155 , -0.73328636,
        0.06003505, -0.00917733,  0.88549679,  0.34398384, -0.21306887,
       -0.37333941,  0.34128025,  0.90386689,  0.69809935,  0.12249289,
        0.43545855, -0.43044596, -0.59102389,  0.04712769,  0.89300558,
       -0.36702834, -0.71436707, -0.43796071,  0.52051314,  0.35770509,
        0.91380035, -0.12726461,  0.70654369,  0.03898736,  0.77268862,
        0.19266253, -0.05369097, -0.31927727,  0.31049791, -0.87384058,
        0.98601238, -0.73352189, -0.05866876, -0.53891623, -0.43880479,
        0.00849503,  0.04674604, -0.46558613, -0.91915535,  0.62952634,
        0.20637471,  0.70979982, -0.53320496, -0.62439568, -0.40624884,
       -0.93315639, -0.602239  ,  0.96386793, -0.39946393, -0.23313713,
       -0.15887892,  0.3761952 , -0.6899759 , -0.75015458, -0.02643102,
       -0.54427453, -0.43309832, -0.38281741, -0.32565884,  0.15682227,
       -0.78782098, -0.37570338, -0.40042479, -0.96805344, -0.10531004,
        0.56318027, -0.09822824, -0.17875489,  0.10915103,  0.03956822,
       -0.5508093 , -0.35232805,  0.15837117, -0.11835377, -0.08241958,
       -0.45869762, -0.5880613 ,  0.7827262 , -0.90316831,  0.14747283,
       -0.88840943,  0.78747163,  0.29776095, -0.45077976, -0.51757493,
        0.27735722, -0.22376483,  0.03642884,  0.10206916, -0.11459233,
       -0.17079386,  0.62589957, -0.01661187,  0.87165391,  0.78365149,
        0.98815448, -0.36501477, -0.07274596,  0.79620318, -0.82733704,
       -0.36854251,  0.08836813,  0.26721806, -0.09493031, -0.05189325,
        0.30089068,  0.05890334,  0.59257191,  0.959237  ,  0.53648339,
       -0.67900334, -0.95651255, -0.49667641, -0.5807751 , -0.1079691 ,
       -0.36422578, -0.68398276,  0.71733788,  0.1362583 , -0.48440539,
        0.48273796, -0.31969288, -0.20921785,  0.2614945 , -0.73158719,
        0.26096645,  0.04822484,  0.09720699,  0.7801111 , -0.77493701,
       -0.88473811,  0.56030816, -0.32724554, -0.085597  , -0.71986467,
       -0.30216631, -0.67605804,  0.85901153,  0.6416126 , -0.96288037,
        0.87184064, -0.37823024,  0.03111141,  0.03708872,  0.03529989,
       -0.26605892, -0.5680592 , -0.78339216, -0.97799889,  0.55850645,
        0.98516099,  0.05394196,  0.26569988,  0.82475726, -0.85693976,
        0.5857472 , -0.66052989,  0.56985387,  0.21923139,  0.44062514,
        0.93821915,  0.34507933, -0.02371641, -0.58033437,  0.69348933,
        0.31221432,  0.12917291,  0.30939278,  0.77759155, -0.82014225,
        0.6393404 ,  0.00866054, -0.59969355,  0.81253703, -0.37619557,
        0.40114675, -0.372405  ,  0.07977299, -0.43752115, -0.59067731,
        0.53953289,  0.33165108, -0.80896625,  0.0577305 , -0.96651881,
        0.29823547,  0.78002325,  0.95831888,  0.57648944, -0.3401485 ,
       -0.16286033, -0.2545454 ,  0.65564576, -0.5515956 ,  0.40957518,
        0.42114522, -0.8791508 ,  0.9070942 , -0.51429528, -0.72967987,
       -0.00457921, -0.55083341,  0.47687555,  0.84605277, -0.82237125,
        0.55812314, -0.44155767,  0.60168572, -0.92549692, -0.18869391,
       -0.46079526, -0.31311813,  0.33797556, -0.2121742 , -0.77438812,
       -0.38968425,  0.8704841 , -0.63443928, -0.83241646, -0.75750411,
       -0.07938886,  0.36735592, -0.3456374 ,  0.31234156, -0.96567295,
        0.02973308,  0.34438492,  0.78502114,  0.93845002, -0.75559007,
        0.7642387 , -0.14405076, -0.01565632,  0.84007427,  0.87025569,
        0.91880846,  0.78598975,  0.58861201,  0.42239642, -0.19475695,
       -0.51305392,  0.06347515, -0.56580587,  0.28259565, -0.56150037,
       -0.54334572,  0.83868058,  0.47380129, -0.06392116, -0.54745765,
       -0.45165529, -0.15429021, -0.98360692,  0.13381556,  0.0630088 ,
        0.16442558, -0.57918582, -0.58986608,  0.02437207,  0.48415534,
       -0.58466698, -0.87446369,  0.73932849, -0.81630107,  0.28735832,
        0.3532498 ,  0.19438695,  0.07740583, -0.59403043,  0.24569857,
        0.56863242,  0.43594652, -0.01293479, -0.42974877, -0.73029113,
       -0.67528619,  0.75705089, -0.82740642, -0.06212661, -0.12260064,
        0.5389949 ,  0.03720438, -0.41648298, -0.18604147,  0.00990349,
        0.17275488, -0.55998139,  0.38470635, -0.38359716,  0.02710132,
        0.13461094,  0.56857548,  0.83753466, -0.7290441 ,  0.66827493,
        0.21008476, -0.53869188,  0.90324389, -0.09176834, -0.05998811,
       -0.05877382, -0.31985842,  0.98275249, -0.37549204,  0.27463374,
       -0.92680964,  0.10215975,  0.13027633,  0.95042525, -0.05357331,
        0.41073975, -0.81283092, -0.16539296,  0.37679162, -0.35737786,
       -0.75893673,  0.41188999,  0.93603408, -0.54218773, -0.52223429,
        0.83944336, -0.66443316, -0.66627892, -0.0659547 ,  0.03078681,
        0.12162139, -0.9752887 , -0.00369149, -0.1884952 ,  0.2781791 ,
        0.29165004,  0.38702831,  0.73894763, -0.23413614,  0.88322156,
        0.75363413, -0.0529799 , -0.53415924, -0.39858809, -0.70521301,
       -0.96665417,  0.72445753, -0.68283538,  0.7046047 , -0.00928268,
       -0.82441022,  0.18598064, -0.90760576,  0.51078488, -0.88160966,
       -0.05356673, -0.85822923, -0.83731508,  0.70909641,  0.88365106,
        0.94864324,  0.10219147,  0.3766693 ,  0.96822948,  0.51707573,
        0.03469657, -0.33911738,  0.78028169, -0.69489182,  0.38674709,
       -0.1056492 , -0.07230066,  0.94225345,  0.82230537, -0.35124299,
        0.77847288, -0.5488149 , -0.09039231, -0.87770346, -0.78409931,
        0.24770466,  0.45255587,  0.11444422, -0.09922586, -0.18259705,
        0.74323762,  0.78548618, -0.56314075,  0.67535305, -0.22438699,
        0.33033243,  0.13703347,  0.3142009 ,  0.1199376 ,  0.75405244,
       -0.24102987, -0.49183387, -0.91400819,  0.39768067, -0.01526501,
        0.27981855,  0.65107903,  0.48197227,  0.56141563,  0.96195768,
        0.31567338, -0.29915061, -0.64076873,  0.42300437, -0.08023197,
       -0.50219142,  0.95983532,  0.59039963, -0.0109391 ,  0.36615138,
        0.21930954, -0.51513908,  0.41082038, -0.97033685,  0.00277071,
       -0.45025559])



class NNetwork:
    """Multilayer fully connected neural network of direct propagation"""

    @staticmethod
    def getTotalWeights(*layers):
        return sum([(layers[i]+1)*layers[i+1] for i in range(len(layers)-1)])

    def __init__(self, inputs, *layers):
        self.layers = []        # list of the number of neurons by layer
        self.acts = []          # list of activation functions (by layer)

        # formation of a list of weight matrices for neurons of each layer and a list of activation functions
        self.n_layers = len(layers)
        for i in range(self.n_layers):
            self.acts.append(self.act_relu)
            if i == 0:
                self.layers.append(self.getInitialWeights(layers[0], inputs+1))         # +1 - this is the input for bias
            else:
                self.layers.append(self.getInitialWeights(layers[i], layers[i-1]+1))    # +1 - this is the input for bias

        self.acts[-1] = self.act_tanh     #the last layer has a threshold activation function

    def getInitialWeights(self, n, m):
        return np.random.triangular(-1, 0, 1, size=(n, m))

    def act_relu(self, x):
        x[x < 0] = 0
        return x
    
    def act_leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def act_th(self, x):
        x[x > 0] = 1
        x[x <= 0] = 0
        return x
    
    def act_tanh(self, x):
        return np.tanh(x)
    
    def act_linear(self, x):
        return x

    def get_weights(self):
        return np.hstack([w.ravel() for w in self.layers])

    def set_weights(self, weights):
        off = 0
        for i, w in enumerate(self.layers):
            w_set = weights[off:off+w.size]
            off += w.size
            self.layers[i] = np.array(w_set).reshape(w.shape)

    def predict(self, inputs):
        f = inputs
        for i, w in enumerate(self.layers):
            f = np.append(f, 1.0)       # adding the input value for the bias
            f = self.acts[i](w @ f)

        return f
    
    

try_model = NNetwork(*[6, 32, 16, 2])
try_model.set_weights(weights) 




# Initialization Pygame
pygame.init()

track_points = [
    (100, 150),  # start
    (300, 150),  # Straight section
    (500, 300),  # Smooth right turn
    (500, 500),  # Straight section down
    (300, 650),  # Smooth left turn
    (100, 500),
    (100, 150),
    
]
track_widths = [50] * len(track_points)  # Track width

# Screen Settings
screen_width, screen_height = 1000, 1000
screen = pygame.display.set_mode((screen_width, screen_height))
font = pygame.font.Font(None, 36) # None is used for the standard font, 36 is the font size

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
    x1, y1 = start_pos
    x2, y2 = end_pos
    dl = dash_length

    dx, dy = x2 - x1, y2 - y1
    distance = math.sqrt(dx**2 + dy**2)
    dash_count = int(distance / dl)
    dash_x = dx / dash_count
    dash_y = dy / dash_count

    for i in range(dash_count):
        if i % 2 == 0:
            start_x = x1 + (i * dash_x)
            start_y = y1 + (i * dash_y)
            end_x = start_x + dash_x
            end_y = start_y + dash_y
            pygame.draw.line(surf, color, (start_x, start_y), (end_x, end_y), width)



def draw_track(screen, track):
    for i in range(len(track.points) - 1):
        start_pos = track.points[i]
        end_pos = track.points[i + 1]
        pygame.draw.line(screen, WHITE, start_pos, end_pos, track.widths[i])

start_point = track_points[0]
end_point = track_points[-1]

# The main cycle of the game
track = Track(track_points, track_widths)
car = Car(start_point[0], start_point[1])
finish_line = (track.points[-3][0], int(track.points[-3][1] - track.widths[-3]/2), track.points[-3][0], int(track.points[-3][1] + track.widths[-3]/2))
print(finish_line)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    prev_car_position = car.x, car.y
    car_state = get_car_state(car, track)
    speed_change, angle_change = neural_network_decision(car_state, try_model)
    car.speed += speed_change
    car.angle += angle_change
    car.update()
    new_car_position = car.x, car.y
    if has_finished(new_car_position, prev_car_position, finish_line):
          text = font.render("FINISH!", True, WHITE, BLACK) 
          text_rect = text.get_rect()
          text_rect.center = (500, 500) # Place the text in the center of the screen
          screen.blit(text, text_rect)
          pygame.display.flip()
          print('finish')
          time.sleep(5)
          break
        
    if is_car_off_track(car, track):
          text = font.render("CRASH!", True, WHITE, BLACK) 
          text_rect = text.get_rect()
          text_rect.center = (500, 500) # Place the text in the center of the screen
          screen.blit(text, text_rect)
          pygame.display.flip()
          print('crash')
          time.sleep(5)
          break  
      
    screen.fill(BLACK)
    draw_track(screen, track)
    draw_dashed_line(screen, BLACK, (finish_line[0], finish_line[1]), (finish_line[2], finish_line[3]), 10)
    # draw_obstacles(screen)
    car.draw(screen)
    pygame.display.flip()

    pygame.time.delay(30)

pygame.quit()





