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


RANDOM_SEED = 42
random.seed(RANDOM_SEED)




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

        # speed
        self.speed = max(0, min(self.speed, self.max_speed))
        
        # turns
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
        pygame.draw.polygon(screen, GREEN, [(self.x + math.cos(math.radians(self.angle)) * self.length, self.y + math.sin(math.radians(self.angle)) * self.length),
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


def neural_network_decision(state, model):
    """
    Make a decision based on the current state of the car and the neural network model.
    
    :param car: car object
    :param model: an instance of a trained neural network model
    :return: speed_change, angle_change
    """
    
    # Disabling the calculation of gradients, since we are in the inference mode
    # with torch.no_grad():
        # We pass the state of the car through a neural network
    output = model.predict(state)
    
    # Extracting the desired speed and angle changes from the network output
    speed_change, angle_change = output[0], output[1]
    
    return speed_change, angle_change



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

    if dx == 0 and dy == 0:  # The segment represents a point
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
            return False  # The car is on the track
    return True  # The car is off the track




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






def simulate_drive(network_weights, car, track, max_time=60):
    time_step = 0.1  # The duration of one time step in seconds.
    total_time = 0  # The total simulation time.
    total_distance = 0  # Initializing the total distance traveled
    
    finish_line = (track.points[-1][0], track.points[-1][1] - track.widths[-1]/2, track.points[-1][0], track.points[-1][1] + track.widths[-1]/2)

    simulation_ended_early = False
    while total_time < max_time:
        prev_car_position = car.x, car.y
        car_state = get_car_state(car, track)
        speed_change, angle_change = neural_network_decision(car_state, network_weights)
        car.speed += speed_change
        car.angle += angle_change
        car.update()
        new_car_position = car.x, car.y

        
        if has_finished(new_car_position, prev_car_position, finish_line):
            print('FINISH!!!!!!!')
            break
        
        if is_car_off_track(car, track):
            print('CRASHHHHHH CAR IS OFF TRACK!!!!!!!!!!!')
            simulation_ended_early = True
            break  # We interrupt the simulation if the car has left the track.
            
        print(car.x, car.y)
        total_distance += car.speed * time_step  # Increasing the distance traveled
        total_time += time_step
    
    return total_distance, total_time, simulation_ended_early







def create_checkpoints(track_points, track_widths):
    checkpoints = []
    for i in range(len(track_points)-1):
        p1 = np.array(track_points[i])
        p2 = np.array(track_points[i+1])
        
        # We find the middle of the segment between two points
        mid_point = (p1 + p2) / 2
        direction = p2 - p1
        normal = np.array([-direction[1], direction[0]])  # Normal to the direction
        normal = normal / np.linalg.norm(normal)  # Normalization
        
        # Creating two checkpoint points perpendicular to the direction of the track
        checkpoint_start = mid_point + normal * track_widths[i] / 2
        checkpoint_end = mid_point - normal * track_widths[i] / 2
        checkpoints.append((checkpoint_start, checkpoint_end))
        
    return checkpoints

def check_checkpoint_crossing(car_position, prev_car_position, checkpoints):
    """We check the intersection of the car line and the checkpoint line"""
    car_pos = np.array(car_position)
    prev_car_pos = np.array(prev_car_position)
    
    for checkpoint in checkpoints:
        cp_start, cp_end = checkpoint
        if line_intersection((cp_start[0],cp_start[1], cp_end[0], cp_end[1]), (car_pos[0],car_pos[1], prev_car_pos[0], prev_car_pos[1])):
            return True
    return False


def find_target_point(prev_car_position, new_car_position, track_segments, current_segment_index, width):
    """
    Defines the next target point on the track.
    
    :param car_position: The current position of the car (x, y).
    :param track_segments: A list of track segments, where each segment is a pair of points (beginning, end).
    :param current_segment_index: The index of the current segment that the car is aiming for.
    :return: The next target point and the index of this segment.
    """
    # Getting the current segment
    current_segment = track_segments[current_segment_index]
    target_line = (current_segment[1][0], current_segment[1][1] + width/2, current_segment[1][0], current_segment[1][1] - width/2)
    trajectory = (prev_car_position[0], prev_car_position[1], new_car_position[0], new_car_position[1])
    # Checking if the car has reached the end of the current segment
    # You can use some threshold radius to determine if a point has been "reached".
    # Approximate threshold radius
    if line_intersection(trajectory, target_line) is not None:
        # If reached the end, move on to the next segment
        current_segment_index = (current_segment_index + 1) % len(track_segments)
        current_segment = track_segments[current_segment_index]
    
    # The target point is the endpoint of the current segment
    target_point = current_segment[1]
    
    return target_point, current_segment_index



def calculate_angle(car_position, prev_car_position, target_point):
    """
    Calculate the angle between the direction of movement of the car and the direction to the target point.

    :param car_position: The current position of the car (x, y).
    :param prev_car_position: The previous position of the car (x, y).
    :param target_point: The target point (x, y).
    :return: The angle in degrees between the direction of movement and the direction to the target point.
    """
    # Car motion vector
    movement_vector = np.array(car_position) - np.array(prev_car_position)
    # Vector to the target point
    target_vector = np.array(target_point) - np.array(car_position)

    # Normalization of vectors
    movement_norm = np.linalg.norm(movement_vector)
    target_norm = np.linalg.norm(target_vector)

    if movement_norm == 0 or target_norm == 0:
        return 0  #If one of the vectors is zero, we consider the angle to be 0

    #Calculating the angle between vectors
    dot_product = np.dot(movement_vector, target_vector)
    angle = np.arccos(dot_product / (movement_norm * target_norm))

    # Converting an angle from radians to degrees
    angle_degrees = np.degrees(angle)

    return angle_degrees


# Simulate track
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



# track = Track(track_points, track_widths)
# finish_line = [(track.points[-1][0], track.points[-1][1] - track.widths[-1]/2, track.points[-1][0], track.points[-1][1] + track.widths[-1]/2)]
# car = Car(start_point[0], start_point[1])



NEURONS_IN_LAYERS = [6, 32, 16, 2]               # distribution of the number of neurons by layers (the first value is the number of inputs)
network = NNetwork(*NEURONS_IN_LAYERS)

LENGTH_CHROM = NNetwork.getTotalWeights(*NEURONS_IN_LAYERS)    # the length of the chromosome to be optimized
LOW = -1.0
UP = 1.0
ETA = 20

# constants of the genetic algorithm
POPULATION_SIZE = 50   # the number of individuals in the population
P_CROSSOVER = 0.9       # the probability of crossing
P_MUTATION = 0.1      # the probability of mutation of an individual
MAX_GENERATIONS = 100    # maximum number of generations
HALL_OF_FAME_SIZE = 2

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register("randomWeight", random.uniform, -1.0, 1.0)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomWeight, LENGTH_CHROM)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)



def eaSimpleElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, callback=None):
    """Redesigned eaSimple algorithm with an element of elitism
    """

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if callback:
            callback[0](*callback[1])

    return population, logbook


def generate_start_positions(track):
    positions = []
    for i in range(len(track.points)-1):
        start_point = track.points[i]
        end_point = track.points[i+1]
        width = track.widths[i]

        #Calculating the vector from the current point to the next one
        vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
        vector_length = (vector[0]**2 + vector[1]**2)**0.5

        # The number of starting positions depends on the length of the section
        num_positions = int(vector_length / width)

        for j in range(num_positions):
            # Interpolate the positions between the current and next points
            ratio = j / num_positions
            new_position = (start_point[0] + vector[0] * ratio, start_point[1] + vector[1] * ratio)
            positions.append(new_position)

    return positions

def calculate_off_track_penalty(checkpoints_passed):
    base_penalty = 500  # The basic penalty for leaving the track
    incremental_penalty = 100  # An additional penalty for each checkpoint passed
    return base_penalty + (incremental_penalty * checkpoints_passed)


#func for training
def getScore(individual):
    network.set_weights(individual)
    
    track = Track(track_points, track_widths)
    start_positions = generate_start_positions(track)
    start_pos = random.choice(track.points)
    # start_pos = track_points[0] 
    car = Car(*start_pos)
    if start_pos == track.points[-2]:
        car.angle = -90
    if start_pos == track.points[-3]:
        car.angle = -135
    if start_pos == track.points[-4]:
        car.angle = -225
    if start_pos == track.points[-5]:
        car.angle = 90
    if start_pos == track.points[-6]:
        car.angle = 45
    
    time_step = 0.1  # The duration of one time step in seconds.
    total_time = 0  # The total simulation time.
    total_distance = 0  # Initializing the total distance traveled
    off_track_penalty = 0
    checkpoints_reward = 0
    # time_since_last_movement = 0
    # stop_time_threshold = 0.2
    stop_penalty=0
    circular_motion_penalty = 0 
    last_checkpoint_distance = 0.0  # Distance to the last checkpoint
    checkpoints_passed_for_penalty = 0 
    checkpoints_passed_for_vector = 1
    
    circular_motion_threshold = 400
    restarts_count = 0  # Restart counter
    
    # We calculate the penalty of L2 regularization
    lambda_l2 = 0.001  # Regularization coefficient
    
    checkpoints = create_checkpoints(track.points, track.widths)
    track_segments = [(track_points[i], track_points[i+1]) for i in range(len(track_points)-1)]
    segment_index = 0
    
    # finish_line = (track.points[-1][0], track.points[-1][1] - track.widths[-1]/2, track.points[-1][0], track.points[-1][1] + track.widths[-1]/2)
    while total_time<750:
        
        prev_car_position = car.x, car.y
        car_state = get_car_state(car, track)

        # print(car_state)
        speed_change, angle_change = neural_network_decision(car_state, network)
        car.speed += speed_change
        car.angle += angle_change
        car.update()
        new_car_position = car.x, car.y
        
        # target_point, segment_index = find_target_point(prev_car_position,new_car_position, track_segments, segment_index, 50)
        
        
        # Проверка на остановку
        if np.linalg.norm(np.array(new_car_position) - np.array(prev_car_position)) < 0.01:  # Почти не двигался
            stop_penalty -= 500   # Accumulation of penalties for lack of movement
            
        # # Checking for the direction of movement
        # if calculate_angle(prev_car_position, new_car_position, target_point) > 90:
        #     dist_mult = 10.0
        # else:
        #     dist_mult = -1.0
            
        
        # Checking for circular motion without passing checkpoints
        distance_travelled = np.linalg.norm(np.array(new_car_position) - np.array(start_pos)) - last_checkpoint_distance
        if distance_travelled > circular_motion_threshold and checkpoints_passed_for_penalty == 0:
            circular_motion_penalty -= 200  #Penalty for roundabout without making progress
            # We reset the distance to prevent the re-imposition of a fine without changing behavior
            last_checkpoint_distance = np.linalg.norm(np.array(new_car_position) - np.array(start_pos))
            
        
        if check_checkpoint_crossing(new_car_position, prev_car_position, checkpoints):
            checkpoints_reward += 1000
            checkpoints_passed_for_penalty += 1  # Increasing the checkpoint counter
            checkpoints_passed_for_vector += 1
            if checkpoints_passed_for_vector == 6:
                checkpoints_passed_for_vector = 0
            last_checkpoint_distance = np.linalg.norm(np.array(new_car_position) - np.array(start_pos))  # Обновляем расстояние до последней контрольной точки

    
        
        # if has_finished(new_car_position, prev_car_position, finish_line):
        #     checkpoints_reward += 100  # A great reward for reaching the finish line
        #     print('FINISH!!!')
        #     break
        
        if is_car_off_track(car, track):
            off_track_penalty -= calculate_off_track_penalty(checkpoints_passed_for_penalty)  # Штраф за выход за пределы трассы
            restarts_count += 1
            # start_pos = random.choice(start_positions)
            # car.reset(*start_pos)  # Restarting from a new random starting position
            break
        # if total_time%40 == 0:
        #     print(f'prev {prev_car_position} \n new {new_car_position} \n target {target_point} \n mult {dist_mult}')
        # print(car.x, car.y)
        total_distance += np.linalg.norm(np.array(new_car_position) - np.array(prev_car_position))  # Увеличиваем пройденное расстояние
        total_time += time_step
    
    l2_penalty = lambda_l2 * network.getTotalWeights(*NEURONS_IN_LAYERS)
    # print('total_distance', total_distance)
    # print('checkpoints_reward', checkpoints_reward)
    # print('stop_penalty', stop_penalty)
    restarts_penalty = -150 * restarts_count 
    fitness = total_distance  + checkpoints_reward + stop_penalty + off_track_penalty + circular_motion_penalty + restarts_penalty - l2_penalty #- (total_time * 0.5)
    print(fitness)
    return fitness,
#%%

# Training
toolbox.register("evaluate", getScore)
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
# toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

population, logbook = eaSimpleElitism(population, toolbox,
                                        cxpb=P_CROSSOVER,
                                        mutpb=P_MUTATION,
                                        ngen=MAX_GENERATIONS,
                                        halloffame=hof,
                                        stats=stats,
                                        verbose=True)


#%%
maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")


best = hof.items[0]
print(best)

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Max/Average fitness')
plt.title('Dependence of maximum and average fitness on generation')
plt.show()
#%%
print(network.get_weights())
print(best)
try_model = NNetwork(*NEURONS_IN_LAYERS)
try_model.set_weights(best)
try_model.get_weights()







#%%
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
font = pygame.font.Font(None, 36)  # None is used for the standard font, 36 is the font size

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def draw_track(screen, track):
    for i in range(len(track.points) - 1):
        start_pos = track.points[i]
        end_pos = track.points[i + 1]
        pygame.draw.line(screen, WHITE, start_pos, end_pos, track.widths[i])

start_point = track_points[0]
end_point = track_points[-1]

# The main cycle of the game
track = Track(track_points, track_widths)
# car = Car(start_point[0], start_point[1])
# car.angle = -90

start_pos = track.points[6]
# start_pos = track_points[0] 
car = Car(*start_pos)
if start_pos == track.points[-2]:
    car.angle = -90
if start_pos == track.points[-3]:
    car.angle = -135
if start_pos == track.points[-4]:
    car.angle = -225
if start_pos == track.points[-5]:
    car.angle = 90
if start_pos == track.points[-6]:
    car.angle = 45
finish_line = (track.points[-1][0], track.points[-1][1] - track.widths[-1]/2, track.points[-1][0], track.points[-1][1] + track.widths[-1]/2)
print(finish_line)
checkpoints = create_checkpoints(track.points, track.widths)
track_segments = [(track_points[i], track_points[i+1]) for i in range(len(track_points)-1)]
segment_index = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    prev_car_position = car.x, car.y
    # print(prev_car_position)
    car_state = get_car_state(car, track)
    speed_change, angle_change = neural_network_decision(car_state, try_model)
    # print('angel', angle_change)
    car.speed += speed_change
    # print('speed', car.speed)
    car.angle += angle_change
    car.update()
    new_car_position = car.x, car.y
    target_point, segment_index = find_target_point(prev_car_position, new_car_position, track_segments, segment_index, 50)
    print(segment_index)
    print(target_point)
    # print(prev_car_position, new_car_position)
    # print(new_car_position)
    if check_checkpoint_crossing(new_car_position, prev_car_position, checkpoints):
        print(car.angle)
    # if has_finished(new_car_position, prev_car_position, finish_line):
    #       text = font.render("FINISH!", True, WHITE, BLACK) 
    #       text_rect = text.get_rect()
    #       text_rect.center = (500, 500) # Place the text in the center of the screen
    #       screen.blit(text, text_rect)
    #       pygame.display.flip()
    #       print('finish')
    #       time.sleep(5)
    #       break
        
    if is_car_off_track(car, track):
          text = font.render("CRASH!", True, WHITE, BLACK) 
          text_rect = text.get_rect()
          text_rect.center = (500, 500) # place the text in the center of the screen
          screen.blit(text, text_rect)
          pygame.display.flip()
          print('crash')
          print(car.angle)
          time.sleep(5)
          break  

    screen.fill(BLACK)
    draw_track(screen, track)
    # draw_obstacles(screen)
    car.draw(screen)
    pygame.display.flip()

    pygame.time.delay(30)

pygame.quit()












