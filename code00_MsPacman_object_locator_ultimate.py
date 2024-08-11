## Object detection in images BY ITS COLOR!

## current state: 
# Detects and separates by color
# After some processing, we get back the location in each frame by pixel indices

# from PIL import Image
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 10)
import webcolors
import os, glob
import scipy.signal as signal
import copy
import pytesseract
import time
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
from itertools import chain

digits_thru_pixels = np.array([
		[[  0,   0,   0,   0,   0,   0,   0,   0,   0],
		[  0,   0,   0, 150, 150, 150,   0,   0,   0],
		[  0,   0, 150, 150,   0, 150, 150,   0,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0,   0, 150, 150,   0, 150, 150,   0,   0],
		[  0,   0,   0, 150, 150, 150,   0,   0,   0],
		[  0,   0,   0,   0,   0,   0,   0,   0,   0]],

		[[  0,   0,   0,   0,   0,   0,   0,   0,   0],
		[  0,   0,   0,   0, 150, 150,   0,   0,   0],
		[  0,   0,   0, 150, 150, 150,   0,   0,   0],
		[  0,   0,   0,   0, 150, 150,   0,   0,   0],
		[  0,   0,   0,   0, 150, 150,   0,   0,   0],
		[  0,   0,   0,   0, 150, 150,   0,   0,   0],
		[  0,   0,   0,   0, 150, 150,   0,   0,   0],
		[  0,   0, 150, 150, 150, 150, 150, 150,   0],
		[  0,   0,   0,   0,   0,   0,   0,   0,   0]],

		[[  0,   0,   0,   0,   0,   0,   0,   0,   0],
		[  0,   0, 150, 150, 150, 150, 150,   0,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0,   0,   0,   0,   0,   0, 150, 150,   0],
		[  0,   0,   0, 150, 150, 150, 150,   0,   0],
		[  0, 150, 150, 150,   0,   0,   0,   0,   0],
		[  0, 150, 150,   0,   0,   0,   0,   0,   0],
		[  0, 150, 150, 150, 150, 150, 150, 150,   0],
		[  0,   0,   0,   0,   0,   0,   0,   0,   0]],

		[[  0,   0,   0,   0,   0,   0,   0,   0,   0],
		[  0, 150, 150, 150, 150, 150, 150,   0,   0],
		[  0,   0,   0,   0,   0,   0, 150, 150,   0],
		[  0,   0,   0,   0,   0,   0, 150, 150,   0],
		[  0,   0, 150, 150, 150, 150, 150,   0,   0],
		[  0,   0,   0,   0,   0,   0, 150, 150,   0],
		[  0,   0,   0,   0,   0,   0, 150, 150,   0],
		[  0, 150, 150, 150, 150, 150, 150,   0,   0],
		[  0,   0,   0,   0,   0,   0,   0,   0,   0]],

		[[  0,   0,   0,   0,   0,   0,   0,   0,   0],
		[  0, 150, 150,   0,   0, 150, 150,   0,   0],
		[  0, 150, 150,   0,   0, 150, 150,   0,   0],
		[  0, 150, 150,   0,   0, 150, 150,   0,   0],
		[  0, 150, 150, 150, 150, 150, 150, 150,   0],
		[  0,   0,   0,   0,   0, 150, 150,   0,   0],
		[  0,   0,   0,   0,   0, 150, 150,   0,   0],
		[  0,   0,   0,   0,   0, 150, 150,   0,   0],
		[  0,   0,   0,   0,   0,   0,   0,   0,   0]],

		[[  0,   0,   0,   0,   0,   0,   0,   0,   0],
		[  0, 150, 150, 150, 150, 150, 150,   0,   0],
		[  0, 150, 150,   0,   0,   0,   0,   0,   0],
		[  0, 150, 150,   0,   0,   0,   0,   0,   0],
		[  0, 150, 150, 150, 150, 150, 150,   0,   0],
		[  0,   0,   0,   0,   0,   0, 150, 150,   0],
		[  0,   0,   0,   0,   0,   0, 150, 150,   0],
		[  0, 150, 150, 150, 150, 150, 150,   0,   0],
		[  0,   0,   0,   0,   0,   0,   0,   0,   0]],

		[[  0,   0,   0,   0,   0,   0,   0,   0,   0],
		[  0,   0, 150, 150, 150, 150, 150,   0,   0],
		[  0, 150, 150,   0,   0,   0,   0,   0,   0],
		[  0, 150, 150,   0,   0,   0,   0,   0,   0],
		[  0, 150, 150, 150, 150, 150, 150,   0,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0,   0, 150, 150, 150, 150, 150,   0,   0],
		[  0,   0,   0,   0,   0,   0,   0,   0,   0]],

		[[  0,   0,   0,   0,   0,   0,   0,   0,   0],
		[  0, 150, 150, 150, 150, 150, 150, 150,   0],
		[  0,   0,   0,   0,   0,   0, 150, 150,   0],
		[  0,   0,   0,   0,   0, 150, 150,   0,   0],
		[  0,   0,   0,   0, 150, 150,   0,   0,   0],
		[  0,   0,   0,   0, 150, 150,   0,   0,   0],
		[  0,   0,   0, 150, 150,   0,   0,   0,   0],
		[  0,   0,   0, 150, 150,   0,   0,   0,   0],
		[  0,   0,   0,   0,   0,   0,   0,   0,   0]],

		[[  0,   0,   0,   0,   0,   0,   0,   0,   0],
		[  0,   0, 150, 150, 150, 150, 150,   0,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0,   0, 150, 150, 150, 150, 150,   0,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0,   0, 150, 150, 150, 150, 150,   0,   0],
		[  0,   0,   0,   0,   0,   0,   0,   0,   0]],

		[[  0,   0,   0,   0,   0,   0,   0,   0,   0],
		[  0,   0, 150, 150, 150, 150, 150,   0,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0, 150, 150,   0,   0,   0, 150, 150,   0],
		[  0,   0, 150, 150, 150, 150, 150, 150,   0],
		[  0,   0,   0,   0,   0,   0, 150, 150,   0],
		[  0,   0,   0,   0,   0,   0, 150, 150,   0],
		[  0,   0, 150, 150, 150, 150, 150,   0,   0],
		[  0,   0,   0,   0,   0,   0,   0,   0,   0]] ])

digits_thru_pixels = np.array(digits_thru_pixels/150, dtype = int)

## ------------ GENERAL PURPOSE ---------------------------

def npwhere_RR(array_x, match_to_be_found):
	array_x = np.array(array_x)
	if ((type(match_to_be_found) == int) | (type(match_to_be_found) == float)): to_return = [i for i in range(array_x.shape[0]) if (array_x[i] == match_to_be_found)]
	else:
		match_to_be_found = np.array(match_to_be_found)
		to_return = []
		for j in range(match_to_be_found.shape[0]):
			to_return = to_return + [i for i in range(array_x.shape[0]) if (array_x[i] == match_to_be_found[j])]

	to_return = np.array(to_return)
	return(to_return)

def show(img):
	plt.imshow(img, cmap="gray")
	plt.show()

def rgbtohex(color_rgb): return "#{:02x}{:02x}{:02x}".format(int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))


# -------------- COLOR SEPARATING OBJECTS -----------------------

def process_image_rgb(image_rgb):
	# Input: Takes the rgb image
	# Returns: (1) a pixel wise (r,g,b) version of the image --> convenient for my calculations
		#(2) all objects separated by color and stored in binary form

	pixel_wise_image = np.zeros([image_rgb.shape[0]*image_rgb.shape[1], image_rgb.shape[2]])
	for i in range(3): pixel_wise_image[:, i] = image_rgb[:,:,i].flatten()

	# Faster than np.unique()
	PWI_T = pixel_wise_image.T
	unique_colors = np.array(list(set(zip(PWI_T[0], PWI_T[1], PWI_T[2]))))

	image_color_separated = [(image_rgb == unique_colors[i]).all(axis = 2) for i in range(unique_colors.shape[0])]
	image_binary = np.array(image_color_separated, dtype = int)	
	# pixel_wise_image = np.zeros([image_rgb.shape[0]*image_rgb.shape[1], image_rgb.shape[2]])
	# for i in range(3): pixel_wise_image[:, i] = image_rgb[:,:,i].flatten()

	# unique_colors_old = np.unique(pixel_wise_image, axis = 0)
	# print(unique_colors.shape[0] - unique_colors_old.shape[0])

	return(pixel_wise_image, image_binary, image_color_separated, unique_colors)


# -------------- DYNAMIC OBJECTS (PACMAN, GHOSTS, FRUITS)  -------------------

def parse_dynamic_objects(image_binary):
	## Locates and parses the dynamic objects TOGETHER: Ms.Pacman, Ghosts and Bonus fruits

	pixel_counts = [np.sum(image_binary[i]) for i in range(image_binary.shape[0])] # Count pixels in each binary image in image_binary
	pixel_counts_sorted = np.sort(pixel_counts)[::-1] # Sort, because we want the max two
	index_max = np.where(pixel_counts == pixel_counts_sorted[0]); index_max_2nd = np.where(pixel_counts == pixel_counts_sorted[1])

	image_antiobjects = image_binary[index_max] + image_binary[index_max_2nd]; image_antiobjects = image_antiobjects[0]
	image_objects_binary = np.array(image_antiobjects == 0, dtype = int) # y: Objects will be 1s in a sea of zeros in this matrix
	return(image_objects_binary, index_max[0][0], index_max_2nd[0][0])


def crop_dyn_objects_in_img_rgb(image_rgb, image_objects_binary):
	object_indices = np.where(image_objects_binary != 0)
	image_objects_rgb = np.zeros_like(image_rgb) # Create empty array like the image.
	image_objects_rgb[object_indices] = image_rgb[object_indices] # Only save the object pixels
	return(image_objects_rgb)


def estimate_point_location_of_objects(array_location_x, array_location_y):
	# Takes in: an object (ghost or pacman)'s (x,y) pixel locations and
	# Returns: midpoint of the enclosing rectangle --> i.e., object's point location (a pixel) estimate
	num_objects = len(array_location_x)
	point_x = np.array([int(np.ceil( ( np.min(array_location_x[i]) + np.max(array_location_x[i]))/2) ) for i in range(num_objects)])
	point_y = np.array([int(np.ceil( ( np.min(array_location_y[i]) + np.max(array_location_y[i]))/2) ) for i in range(num_objects)])
	return(point_x, point_y)


def estimate_point_location_of_a_single_object(image_single_object_binary):
	# Takes in: an object (ghost or pacman)'s (x,y) pixel locations and
	# Returns: midpoint of the enclosing rectangle --> i.e., object's point location (a pixel) estimate
	object_indices = np.where(image_single_object_binary != 0)
	array_location_x = object_indices[0]
	array_location_y = object_indices[1]
	if (len(array_location_x !=0)):
		point_x = int(np.ceil( ( np.min(array_location_x) + np.max(array_location_x))/2) )
		point_y = int(np.ceil( ( np.min(array_location_y) + np.max(array_location_y))/2) )
	else: point_x, point_y = 0, 0
	return(point_x, point_y)


def separate_scared_ghosts(image_objects):
	objects_lumped = np.nonzero(image_objects) # all locations of 1s (i.e., objects)
	temp = np.where(np.diff(objects_lumped[0])>1) # identifying separation points of different objects
	objects_x = np.split(objects_lumped[0], temp[0] + 1) # Each object's x coordinate pixels separated 
	objects_y = np.split(objects_lumped[1], temp[0] + 1) # Each object's y coordinate pixels separated

	num_dynamic_objects = len(objects_x)
	pixel_per_object = [objects_x[i].shape[0] for i in range(num_dynamic_objects)]

	point_location_x, point_location_y = estimate_point_location_of_objects(array_location_x = objects_x, array_location_y = objects_y)
	return(objects_x, objects_y, point_location_x, point_location_y, pixel_per_object)


# def locate_and_identify_dynamic_objects(image_binary): # **************OBSOLETE*****************
# 	## Locates and identifies the dynamic objects: Ms.Pacman, Ghosts and Bonus fruits

# 	pixel_counts = [np.sum(image_binary[i]) for i in range(image_binary.shape[0])] # Count pixels in each binary image in image_binary
# 	pixel_counts_sorted = np.sort(pixel_counts)[::-1] # Sort, because we want the max two
# 	index_max = np.where(pixel_counts == pixel_counts_sorted[0]); index_max_2nd = np.where(pixel_counts == pixel_counts_sorted[1])

# 	image_antiobjects = image_binary[index_max] + image_binary[index_max_2nd]; image_antiobjects = image_antiobjects[0]
# 	image_objects = np.array(image_antiobjects == 0, dtype = int) # y: Objects will be 1s in a sea of zeros in this matrix
# 	image_objects = image_objects[1:172] # We crop to only the main gameplay area.

# 	objects_lumped = np.nonzero(image_objects) # all locations of 1s (i.e., objects)
# 	temp = np.where(np.diff(objects_lumped[0])>1) # identifying separation points of different objects
# 	objects_x = np.split(objects_lumped[0], temp[0] + 1) # Each object's x coordinate pixels separated 
# 	objects_y = np.split(objects_lumped[1], temp[0] + 1) # Each object's y coordinate pixels separated

# 	num_dynamic_objects = len(objects_x)
# 	pixel_per_object = [objects_x[i].shape[0] for i in range(num_dynamic_objects)]


# 	point_location_x, point_location_y = estimate_point_location_of_objects(array_location_x = objects_x, array_location_y = objects_y)
# 	return(image_objects, objects_x, objects_y, point_location_x, point_location_y, pixel_per_object)


# -------------- STATIC OBJECTS (DOTS AND PILLS)  -------------------

def count_locate_and_remove_static_objects(image_binary_one_layer, static_object_type):
	if (static_object_type == 'dot'): length_object, width_object = 2, 4
	elif (static_object_type == 'pill'): length_object, width_object = 7, 4

	def_object = np.zeros([length_object + 2, width_object + 2]).astype(int)
	def_object[1: length_object+1, 1: width_object+1] = 1

	# empty spaces need to be -1 for signal.correlate (instead of 0)
	image_binary_one_layer[image_binary_one_layer == 0] = -1
	def_object[def_object == 0] = -1

	max_peak = np.prod(def_object.shape)
	c = signal.correlate(image_binary_one_layer, def_object, 'valid'); c = c.astype(int)
	overlaps = np.array(np.where(c == max_peak))

	image_binary_one_layer[image_binary_one_layer == -1] = 0
	
	if (overlaps.size != 0):
	# Case: we found at least one matching object!
		object_count = overlaps.shape[1]
		objects_point_locx = overlaps[0] + 1
		objects_point_locy = overlaps[1] + 1
		
		image_binary_maze_objects_removed = copy.copy(image_binary_one_layer)
		for i,j in zip(overlaps[0],overlaps[1]):
			image_binary_maze_objects_removed[i+1:i+(length_object+1),j+1:j+(width_object+1)] = 0
		image_binary_maze_objects_removed = np.array(image_binary_maze_objects_removed == 0, dtype = int) # Pathways will be 1s in a sea of zeros (the walls) in this matrix
	
	else:
	# Case: No matching object
		object_count, objects_point_locx, objects_point_locy, image_binary_maze_objects_removed = \
		0, np.array([0]), np.array([0]), np.array(image_binary_one_layer == 0, dtype = int)
	return(object_count, objects_point_locx, objects_point_locy, image_binary_maze_objects_removed)

# -------------- SCORE  -------------------

# Obsolete for Pacman
# def ocr_score_old(img_rgb): ## With pytesseract
# 	img_grayscale = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# 	img_thresh = cv2.threshold(img_grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# 	custom_oem_psm_config = r'-l eng --tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata" --oem 2 --psm 6 outputbase digits -c tessedit_char_whitelist=0123456789'
# 	score = pytesseract.image_to_string(img_thresh, config=custom_oem_psm_config)
# 	score = int(score)
# 	return(score)

def ocr_score(img_rgb):
	img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
	# show(img_gray)

	## We will read the digits last to first. I know the actual location on the img, for each digit.
	## The consistency helps to easily identify the digits through the following
	
	img_digit = [-1]*6 # Placeholder for each digit img
	each_digit = [-1]*6
	read_score = 0 # To sum things down at the end, to just have the numerical score.

	# 186:194 sets the rows and the column for the rightmost digit is 94:103
	# Then, each digit can be located 8 pixels apart (columnwise).
	for digit_i in range(6): 
		img_digit[digit_i] = img_gray[186:195,94 - 8*digit_i : 103 - 8*digit_i]

	for i in range(len(img_digit)):
		img_one_digit = img_digit[i]
		unique_color_numbers = np.unique(img_one_digit)
		# If there isn't actually a number	
		if (img_one_digit.sum() <5): break
		# OR... If we fail to read the number for transitions, 
			# we should abandon the whole score. We can fill these with previous filled entry
		elif (unique_color_numbers.shape[0]>2): 
			read_score = -1
			break
		else:
			# Else, read the digit by matching defined pixelated digits
			img_one_digit = np.array(img_one_digit > 0, dtype = int)
			read_one_digit = [i for i in range(len(digits_thru_pixels)) if (img_one_digit == digits_thru_pixels[i]).all()]
			if (len(read_one_digit) >0): 
				read_one_digit = read_one_digit[0]
				read_score = read_score + int(read_one_digit)*(10**i)
			else: read_score = np.nan

		# elif(unique_color_numbers.shape[0] >2):
		# 	different_color_numbers = unique_color_numbers[unique_color_numbers != 0]

		
		
	# print(read_score)
	return(read_score)

# --------------- NUM LIVES LEFT -------------------
# --------------------------------------------------

def count_num_lives_left(img_binary, unique_colors_hex):

	life_color_hex = "#bbbb35"
	if ((np.array(unique_colors_hex) == life_color_hex).any()):
		life_color_index = np.where(np.array(unique_colors_hex) == life_color_hex)[0][0]
		if (img_binary[life_color_index][175:190, 25:40].sum() > 2):
			num_life_left = 2
		else:
			if (img_binary[life_color_index][175:190, :25].sum() > 2):
				num_life_left = 1
	else: num_life_left = 0
	return(num_life_left)

# --------------------------------------------------
# -------------- ALL ABOUT MAZES -------------------
# --------------------------------------------------

def load_maze():
	## Loading mazes for each level present in the dataset.
	# In this dataset 3 levels in 5 rounds.
	maze_list = ['under_the_hood_things_by_RR/five_round_layouts/Level_1_Round_1_2_with_pathlines.png', 
				 'under_the_hood_things_by_RR/five_round_layouts/Level_2_Round_3_4_with_pathlines.png', \
				 'under_the_hood_things_by_RR/five_round_layouts/Level_3_Round_5_with_pathlines.png'] # glob.glob("five_round_layouts//*.png")
	num_levels = len(maze_list)
	num_rounds = 5
	pathway_color_hex = '#22b14c'

	image_rgb_maze_by_level = [0]*num_levels
	image_binary_maze_by_level = [0]*num_levels
	image_binary_pathway_by_level = [0]*num_levels

	for maze_i in range(num_levels):
		image_name = maze_list[maze_i]
		# Here the image is being read in "BGR" format, and immediately converted to RGB
		image_rgb_maze_by_level[maze_i] = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
		_ , image_binary_temp, _, unique_colors_maze = process_image_rgb(image_rgb = image_rgb_maze_by_level[maze_i])

		unique_colors_hex_maze = [rgbtohex(unique_colors_maze[i]) for i in range(unique_colors_maze.shape[0])]

		# The following is to extract the maze pathway area
		cropped_image = image_binary_temp[:,:172,:]
		pixel_counts = [np.sum(cropped_image[i]) for i in range(cropped_image.shape[0])] # Count pixels in each binary image in image_binary
		pixel_counts_sorted = np.sort(pixel_counts)[::-1] # Sort, because we want the max twp
		index_max_2nd = np.where(pixel_counts == pixel_counts_sorted[1])
		image_binary_maze_by_level[maze_i] = np.array(image_binary_temp[index_max_2nd] == 0, dtype = int)[0]


		# The following is to extract the maze pathway lines
		if ((np.array(unique_colors_hex_maze) == pathway_color_hex).any()):
			pathway_color_index = np.where(np.array(unique_colors_hex_maze) == pathway_color_hex)[0][0]
			image_binary_pathway_by_level[maze_i] = image_binary_temp[pathway_color_index]

	return(image_binary_maze_by_level, image_binary_pathway_by_level)


def identify_maze(image_maze_binary_from_frame, image_binary_maze_by_level):
	match_found = 0
	matching_maze_index = -1

	detected_maze = image_maze_binary_from_frame[1:150,:]
	# show(detected_maze)
	for maze_i in range(len(image_binary_maze_by_level)):
		 defined_maze = image_binary_maze_by_level[maze_i][1:150,:]

		 match_count = (defined_maze == detected_maze).sum()
		 match_ratio = match_count/ (defined_maze.shape[0]*defined_maze.shape[1])
		 if (match_ratio > 0.80):
		 	# print("%f match with maze %i"%(match_ratio, maze_i))
		 	matching_maze_index = maze_i
		 	match_found = 1
	if (match_found == 0): print("WARNING! NO match :( -------------> > > ")
	return(match_found, matching_maze_index)

# In this function, we calculate the distances of all points on maze pathways to the given point.
# Then we use the closest point on the pathway as the 
# How the pathways in the mazes can help?
	# Benefits: 
			# (1) As the ghosts and the pacman can roam along these lines, our computational problem is much more reduced (in other words, a much lower number of total computations are needed).
			# (2) We can treat the lines as a reduced space of movement for the objects. Therefore, their locational information can be represented as distributions in this space.

def project_single_point_on_pathway(img_binary_pathway_by_level, raw_point_locx, raw_point_locy, maze_config_number): # using minimum euclidean distance from point to pathway
	# find the closest point on pathway

	# Background: we had represented the pathways in the mazes as 1s in sea of 0's, essentially as an image with just the pathways as 1 pixel wide lines in an otherwise blank world.
	# We would project the approximated locations on to the pathways to make our approximations more accurate.
	# Here we are saving the x and the y coordinates

	img_maze = img_binary_pathway_by_level[maze_config_number]
	if img_maze[raw_point_locx, raw_point_locy] !=0:
		projected_point_locx, projected_point_locy = raw_point_locx, raw_point_locy
	else:
		x, y =  np.where(img_maze != 0) 
		del_x = x - raw_point_locx
		del_y = y - raw_point_locy
		del_r = np.sqrt(del_x**2 + del_y**2)
		loc_min_del_r = np.argmin(del_r)

		if (np.min(del_r)<10): # This condition ensures that we are taking points within the pathway (and not from absurd areas)
			projected_point_locx, projected_point_locy = raw_point_locx + del_x[loc_min_del_r], raw_point_locy + del_y[loc_min_del_r]
		else:
			projected_point_locx, projected_point_locy = raw_point_locx, raw_point_locy

	# OLD version
	# x_coordinate = del_x =  np.where(img_binary_pathway_by_level[maze_config_number] != 0)[0] - raw_point_locx
	# y_coordinate = del_y =  np.where(img_binary_pathway_by_level[maze_config_number] != 0)[1] - raw_point_locy
	# r = np.sqrt(x_coordinate**2 + y_coordinate**2)
	# # print(np.argmin(r))
	# if (np.min(r)<10): # This condition ensures that we are taking points within the pathway (and not from absurd areas)
	# 	r_min_index = np.where(r == np.min(r))[0][0]
	# 	projected_point_locx = np.where(img_binary_pathway_by_level[maze_config_number] != 0)[0][r_min_index]
	# 	projected_point_locy = np.where(img_binary_pathway_by_level[maze_config_number] != 0)[1][r_min_index]
	# else:
	# 	projected_point_locx = raw_point_locx
	# 	projected_point_locy = raw_point_locy

	## temp
	# point_as_img = np.zeros_like(img_binary_pathway_by_level[maze_config_number])
	# point_as_img[raw_point_locx, raw_point_locy] = 2
	# point_as_img[projected_point_locx, projected_point_locy] = 3
	# print(img_binary_pathway_by_level[maze_config_number])
	# plt.imshow(img_binary_pathway_by_level[maze_config_number] + point_as_img, origin = "lower")
	# # plt.show()
	# print(r.shape)
	return(projected_point_locx, projected_point_locy)

def project_batch_of_points_on_pathway(img_binary_pathway_by_level, raw_point_locx_list, raw_point_locy_list, maze_config_number):
	projected_point_locx_list = [0] * len(raw_point_locx_list)
	projected_point_locy_list = [0] * len(raw_point_locx_list)
	for point_i in range(len(raw_point_locx_list)):
		projected_point_locx_list[point_i], projected_point_locy_list[point_i] = \
			project_single_point_on_pathway(img_binary_pathway_by_level, raw_point_locx_list[point_i], raw_point_locy_list[point_i], maze_config_number)
	projected_point_locx_array = np.array(projected_point_locx_list)
	projected_point_locy_array = np.array(projected_point_locy_list)
	return(projected_point_locx_array, projected_point_locy_array)

def locate_everything(image_name, img_binary_maze_by_level, img_binary_pathway_by_level):
	num_pacman = num_red_ghost = num_orange_ghost = num_purple_ghost = num_cyan_ghost = num_scared_ghosts = num_dots = num_energy_pills = \
	pacman_point_locx = red_ghost_point_locx = orange_ghost_point_locx = purple_ghost_point_locx = cyan_ghost_point_locx = \
	pacman_point_locy = red_ghost_point_locy = orange_ghost_point_locy = purple_ghost_point_locy = cyan_ghost_point_locy = \
	fruit_type_by_color = fruit_point_locx = fruit_point_locy = score = -1

	pacman_point_locx_projected = red_ghost_point_locx_projected = orange_ghost_point_locx_projected = purple_ghost_point_locx_projected = cyan_ghost_point_locx_projected = \
	pacman_point_locy_projected = red_ghost_point_locy_projected = orange_ghost_point_locy_projected = purple_ghost_point_locy_projected = cyan_ghost_point_locy_projected = \
	fruit_point_locx_projected = fruit_point_locy_projected = -1

	maze_config_number = -100
	scared_ghosts_point_locx = dots_point_locx = energy_pills_point_locx = scared_ghosts_point_locy = dots_point_locy = energy_pills_point_locy = np.array([-1,-1])
	scared_ghosts_point_locx_projected = dots_point_locx_projected = energy_pills_point_locx_projected = scared_ghosts_point_locy_projected = \
	dots_point_locy_projected = energy_pills_point_locy_projected = np.array([-1,-1])
	img_rgb = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

	if img_rgb is None:
		sys.exit("Could not read the image. Dun dun dun...")

	time_distribution = []
	time_distribution.append(time.time())
## --------------------- STEP: RAW IMAGE --> JUST THE OBJECTS ---------------- 
	# In the next block, we will crop out just the dynamic objects in an rgb
	# This way set the stage for next step of processing
	_, img_binary_orig, _, _ = process_image_rgb(image_rgb = img_rgb)
	img_objects_binary, index_max, index_max_2nd = parse_dynamic_objects(image_binary = img_binary_orig)
	img_objects_rgb = crop_dyn_objects_in_img_rgb(image_rgb = img_rgb, image_objects_binary = img_objects_binary)
	time_distribution.append(time.time())
## ------------------ STEP: IMG of ONLY OBJECTS ---> OBJECTS SEPARATED --------
	_ , img_binary, image_color_separated, unique_colors = process_image_rgb(image_rgb = img_objects_rgb)
	
	unique_colors = np.array(unique_colors, dtype = int)
	unique_colors_hex = ['#%02x%02x%02x' %tuple(unique_colors[i]) for i in range(unique_colors.shape[0])]
	time_distribution.append(time.time())
# -------------------------------------------------------------------------------------------------------------------------

# --------------------- STEP: Locate Pacman --------------------------------------
	pacman_color_hex_list = ["#d2a44a", "#8e8e8e"] # if the first has a match, no need to search for next items.

	for pacman_color_hex in pacman_color_hex_list:
		if ((np.array(unique_colors_hex) == pacman_color_hex).any()):
			num_pacman = 1
			pacman_color_index = np.where(np.array(unique_colors_hex) == pacman_color_hex)[0][0]
			pacman_point_locx, pacman_point_locy = estimate_point_location_of_a_single_object(img_binary[pacman_color_index])
			# print("found pacman")
			break

## Non-scared ghosts of different colors

# --------------------- STEP: Locate Red_ghost --------------------------------------
	# red_ghost_color_hex = "#c84848"
	red_ghost_color_hex_list = ["#c84848", "#97197a"] # if the first has a match, no need to search for next items.

	for red_ghost_color_hex in red_ghost_color_hex_list:
		if ((np.array(unique_colors_hex) == red_ghost_color_hex).any()):
			num_red_ghost = 1
			red_ghost_color_index = np.where(np.array(unique_colors_hex) == red_ghost_color_hex)[0][0]
			red_ghost_point_locx, red_ghost_point_locy = estimate_point_location_of_a_single_object(img_binary[red_ghost_color_index])
			# print("found blinky (red)")
			break

# --------------------- STEP: Locate orange_ghost --------------------------------------
	# orange_ghost_color_hex = "#b47a30"
	orange_ghost_color_hex_list = ["#b47a30", "#6f6f6f"] # if the first has a match, no need to search for next items.

	for orange_ghost_color_hex in orange_ghost_color_hex_list:
		if ((np.array(unique_colors_hex) == orange_ghost_color_hex).any()):
			num_orange_ghost = 1
			orange_ghost_color_index = np.where(np.array(unique_colors_hex) == orange_ghost_color_hex)[0][0]
			orange_ghost_point_locx, orange_ghost_point_locy = estimate_point_location_of_a_single_object(img_binary[orange_ghost_color_index])
			break

# --------------------- STEP: Locate purple_ghost --------------------------------------
	# purple_ghost_color_hex = "#c659b3"
	purple_ghost_color_hex_list = ["#c659b3", "#9246c0"] # if the first has a match, no need to search for next items.

	for purple_ghost_color_hex in purple_ghost_color_hex_list:
		if ((np.array(unique_colors_hex) == purple_ghost_color_hex).any()):
			num_purple_ghost = 1
			purple_ghost_color_index = np.where(np.array(unique_colors_hex) == purple_ghost_color_hex)[0][0]
			purple_ghost_point_locx, purple_ghost_point_locy = estimate_point_location_of_a_single_object(img_binary[purple_ghost_color_index])
			break

# --------------------- STEP: Locate cyan_ghost --------------------------------------
	# cyan_ghost_color_hex = "#54b899"
	cyan_ghost_color_hex_list = ["#54b899", "#429e82"] # if the first has a match, no need to search for next items.

	for cyan_ghost_color_hex in cyan_ghost_color_hex_list:
		if ((np.array(unique_colors_hex) == cyan_ghost_color_hex).any()):
			num_cyan_ghost = 1
			cyan_ghost_color_index = np.where(np.array(unique_colors_hex) == cyan_ghost_color_hex)[0][0]
			cyan_ghost_point_locx, cyan_ghost_point_locy = estimate_point_location_of_a_single_object(img_binary[cyan_ghost_color_index])
			break
	time_distribution.append(time.time())
## SCARED GHOSTS of same blue color
# --------------------- STEP: Locate scared_ghosts --------------------------------------
	scared_ghosts_color_hex = ["#4272c2", "#2d57b0"] 
		# IMPORTANT: even if the first has a match, we need to search for next items.
		# This one is more complicated than others, as different ghosts can have different colors on the same frame.
		# I am thinking of combining pixels of each color, before separating the scared ghosts.
		# This way, I may be able to correctly parse all ghosts.

	scared_ghosts_color_hex_available = list(set(scared_ghosts_color_hex) & set(unique_colors_hex))
	if (len(scared_ghosts_color_hex_available) != 0):
		scared_ghosts_color_index = npwhere_RR(unique_colors_hex, scared_ghosts_color_hex_available)
		img_binary_sg = np.zeros([210,160])
		for i in range(len(scared_ghosts_color_hex_available)):
			img_binary_sg = img_binary_sg + img_binary[scared_ghosts_color_index[i]]
		_, _, scared_ghosts_point_locx, scared_ghosts_point_locy, _ = separate_scared_ghosts(img_binary_sg)
		num_scared_ghosts = len(scared_ghosts_point_locx)

	time_distribution.append(time.time())
# --------------------- STEP: Locate fruits on screen by matching the fruit color specified at the bottom ---------------------------
	_, _, _, fruit_color = process_image_rgb(img_objects_rgb[175:185,120:130])
	fruit_color = fruit_color[1]

	fruit_color_hex = rgbtohex(fruit_color)
	if ((np.array(unique_colors_hex) == fruit_color_hex).any()):
		num_fruit = 1
		fruit_color_index = np.where(np.array(unique_colors_hex) == fruit_color_hex)[0][0]
		fruit_type_by_color = fruit_color_hex
		fruit_point_locx, fruit_point_locy = estimate_point_location_of_a_single_object(img_binary[fruit_color_index][:173])

	time_distribution.append(time.time())
## --------------------- STEP: Locating (1) DOTS and (2) ENERGY PILLS + (3) extracting maze from frame
	# ---------------------> dots
	num_dots, dots_point_locx, dots_point_locy, img_maze_binary_from_frame = \
		count_locate_and_remove_static_objects(image_binary_one_layer = img_binary_orig[index_max_2nd], static_object_type = 'dot')
	# ---------------------> energy_pills
	num_energy_pills, energy_pills_point_locx, energy_pills_point_locy, _ = \
		count_locate_and_remove_static_objects(image_binary_one_layer = img_binary_orig[index_max_2nd], static_object_type = 'pill')

# --------------------- STEP: Reading the score through OCR --------------------------------------
	score = ocr_score(img_rgb)
	time_distribution.append(time.time())
# --------------------- STEP: Counting the number of lives left --------------------------------------
	num_life_left = count_num_lives_left(img_binary, unique_colors_hex)
	time_distribution.append(time.time())
# --------------------- STEP: Maze layout identification --------------------------------------

	_, maze_config_number = identify_maze(img_maze_binary_from_frame, img_binary_maze_by_level)
	time_distribution.append(time.time())
# --------------------- STEP: Project all points on maze pathway --------------------------------------
	pacman_point_locx_projected, pacman_point_locy_projected = project_single_point_on_pathway(img_binary_pathway_by_level, pacman_point_locx, pacman_point_locy, maze_config_number)
	red_ghost_point_locx_projected, red_ghost_point_locy_projected = project_single_point_on_pathway(img_binary_pathway_by_level, red_ghost_point_locx, red_ghost_point_locy, maze_config_number)
	orange_ghost_point_locx_projected, orange_ghost_point_locy_projected = project_single_point_on_pathway(img_binary_pathway_by_level, orange_ghost_point_locx, orange_ghost_point_locy, maze_config_number)
	purple_ghost_point_locx_projected, purple_ghost_point_locy_projected = project_single_point_on_pathway(img_binary_pathway_by_level, purple_ghost_point_locx, purple_ghost_point_locy, maze_config_number)
	scared_ghosts_point_locx_projected, scared_ghosts_point_locy_projected = project_batch_of_points_on_pathway(img_binary_pathway_by_level, scared_ghosts_point_locx, scared_ghosts_point_locy, maze_config_number)
	cyan_ghost_point_locx_projected, cyan_ghost_point_locy_projected = project_single_point_on_pathway(img_binary_pathway_by_level, cyan_ghost_point_locx, cyan_ghost_point_locy, maze_config_number)
	fruit_point_locx_projected, fruit_point_locy_projected = project_single_point_on_pathway(img_binary_pathway_by_level, fruit_point_locx, fruit_point_locy, maze_config_number)
	dots_point_locx_projected, dots_point_locy_projected = project_batch_of_points_on_pathway(img_binary_pathway_by_level, dots_point_locx, dots_point_locy, maze_config_number)	
	energy_pills_point_locx_projected, energy_pills_point_locy_projected = project_batch_of_points_on_pathway(img_binary_pathway_by_level, energy_pills_point_locx, energy_pills_point_locy, maze_config_number)	

	time_distribution.append(time.time())
	# print(np.diff(time_distribution), np.sum(np.diff(time_distribution)))
# --------------------- TEMPORARY: Printing and Plotting things -------------------------------------------
	# print(scared_ghosts_point_locx)
	# print(dots_point_locx)
	# TEMP_points_x = [pacman_point_locx, red_ghost_point_locx, orange_ghost_point_locx, purple_ghost_point_locx, cyan_ghost_point_locx, fruit_point_locx]
	# TEMP_points_y = [pacman_point_locy, red_ghost_point_locy, orange_ghost_point_locy, purple_ghost_point_locy, cyan_ghost_point_locy, fruit_point_locy]
	# TEMP_points_x = TEMP_points_x + list(scared_ghosts_point_locx); TEMP_points_y = TEMP_points_y + list(scared_ghosts_point_locy)
	# TEMP_points_x = TEMP_points_x + list(dots_point_locx); TEMP_points_y = TEMP_points_y + list(dots_point_locy)
	# TEMP_points_x = TEMP_points_x + list(energy_pills_point_locx); TEMP_points_y = TEMP_points_y + list(energy_pills_point_locy)

	# TEMP_points_x_projected = [pacman_point_locx_projected, red_ghost_point_locx_projected, orange_ghost_point_locx_projected, purple_ghost_point_locx_projected, cyan_ghost_point_locx_projected, fruit_point_locx_projected]
	# TEMP_points_y_projected = [pacman_point_locy_projected, red_ghost_point_locy_projected, orange_ghost_point_locy_projected, purple_ghost_point_locy_projected, cyan_ghost_point_locy_projected, fruit_point_locy_projected]
	# TEMP_points_x_projected = TEMP_points_x_projected + list(scared_ghosts_point_locx_projected); TEMP_points_y_projected = TEMP_points_y_projected + list(scared_ghosts_point_locy_projected)
	# TEMP_points_x_projected = TEMP_points_x_projected + list(dots_point_locx_projected); TEMP_points_y_projected = TEMP_points_y_projected + list(dots_point_locy_projected)
	# TEMP_points_x_projected = TEMP_points_x_projected + list(energy_pills_point_locx_projected); TEMP_points_y_projected = TEMP_points_y_projected + list(energy_pills_point_locy_projected)

	# # # TEMP_points_x_projected2, TEMP_points_y_projected2 = project_batch_of_points_on_pathway(img_binary_pathway_by_level, TEMP_points_x, TEMP_points_y, maze_config_number)

	# # # TEMP_points_x_projected_alternate_failed, TEMP_points_y_projected_alternate_failed = project_batch_of_points_on_pathway_alternate_failed(TEMP_points_x, TEMP_points_y, maze_config_number)
	# # print(TEMP_points_x_projected)
	# # print("___________________")

	# fig = plt.figure()
	# ax0 = fig.add_subplot(121)
	# ax1 = fig.add_subplot(122)
	# # plt.imshow(img_objects_rgb)
	# image_to_plot = img_binary_pathway_by_level[maze_config_number] + img_objects_binary+ img_binary_maze_by_level[maze_config_number]
	# # image_to_plot = Image.fromarray(image_to_plot).convert("L")
	# ax0.imshow(image_to_plot, cmap='gray')
	# ax0.plot(TEMP_points_y, TEMP_points_x, 'bx')
	# # ax0.plot(TEMP_points_y_projected_alternate_failed, TEMP_points_x_projected_alternate_failed, 'r.')
	# # ax0.plot(TEMP_points_y_projected2, TEMP_points_x_projected2, 'w+')
	# ax0.plot(TEMP_points_y_projected, TEMP_points_x_projected, 'g^')
	# ax0.legend()
	# ax1.imshow(img_rgb)
	# # plt.show()
	# plt.close()	

	# print(scared_ghosts_point_locy_projected)
	# print(energy_pills_point_locx)
	# print(dots_point_locy)

## ----------------- UNOCoMMENT THE WHOLE OF ABOVE AT ONCE (SO THAT SOME DOUBLE COMMENTED THINGS REMAIN COMMENTED) ...
## -------------------------  ....  TO PRINT AND PLOT  ----------------------------

	return(num_pacman, pacman_point_locx, pacman_point_locy, pacman_point_locx_projected, pacman_point_locy_projected, \
		num_red_ghost, red_ghost_point_locx, red_ghost_point_locy, red_ghost_point_locx_projected, red_ghost_point_locy_projected, \
		num_orange_ghost, orange_ghost_point_locx, orange_ghost_point_locy, orange_ghost_point_locx_projected, orange_ghost_point_locy_projected, \
		num_purple_ghost, purple_ghost_point_locx, purple_ghost_point_locy, purple_ghost_point_locx_projected, purple_ghost_point_locy_projected, \
		num_cyan_ghost, cyan_ghost_point_locx, cyan_ghost_point_locy,  cyan_ghost_point_locx_projected, cyan_ghost_point_locy_projected, \
		num_scared_ghosts, scared_ghosts_point_locx, scared_ghosts_point_locy,  scared_ghosts_point_locx_projected, scared_ghosts_point_locy_projected, \
		num_dots, dots_point_locx, dots_point_locy,  dots_point_locx_projected, dots_point_locy_projected, \
		num_energy_pills, energy_pills_point_locx, energy_pills_point_locy,  energy_pills_point_locx_projected, energy_pills_point_locy_projected, \
		fruit_type_by_color, fruit_point_locx, fruit_point_locy, fruit_point_locx_projected, fruit_point_locy_projected, \
		score, maze_config_number, num_life_left, unique_colors_hex)

	# return(img_objects_rgb, img_binary, img_maze_binary_from_frame, maze_config_number)

## --------------------------- First layer of Testing ----------------------------------------

# if __name__ == "__main__":
# 	## Start by loading the mazes
# 	time_start = time.time()
# 	img_binary_maze_by_level, img_binary_pathway_by_level = load_maze()

# 	## Specification (Mostly automated)
# 	image_data_directory = "C:/Users/Roussel/Documents/Atari Research/AtariHead_codes/Test_images_new" ##xx
# 	os.chdir(image_data_directory) # makes the action data directory CURRENT ## xx
# 	## The following gives a list of all files with "*.xyz" extention.
# 	image_list = glob.glob("*.png")


# 	for image_i in range(len(image_list)):
# 		image_name = image_data_directory+ "/" + image_list[image_i]
# 		locate_everything(image_name, img_binary_maze_by_level, img_binary_pathway_by_level)
# 	print(time.time()-time_start)

## ------------------------------------------------------------------------------------------



## --------------------------- Second Layer of testing ----------------------------------------

if __name__ == "__main__":
	## Start by loading the mazes
	time_start = time.time()
	img_binary_maze_by_level, img_binary_pathway_by_level = load_maze()

	## Specification: The images we wanna locate things from.
	image_data_directory = 'extracted_datafiles/105_RZ_3614646_Aug-24-17-47-26'
	image_list = os.listdir(image_data_directory)

	# for i in range(1000): print(project_single_point_on_pathway(img_binary_pathway_by_level, 65, 50, 0))
	# print(time.time() - time_start)
	# exit()
	for image_i in range(10): #len(image_list)):
		image_name = image_data_directory+ "/" + image_list[image_i]
		print(image_name)
		locate_everything(image_name, img_binary_maze_by_level, img_binary_pathway_by_level)
	print(time.time()-time_start)

## ------------------------------------------------------------------------------------------