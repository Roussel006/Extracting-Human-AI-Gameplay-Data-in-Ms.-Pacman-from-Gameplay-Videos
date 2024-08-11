"""
1. Locate all objects
	Pacman, ghotst, dots, pills, fruits, lives

2. Calculate necessary distances
	Distances along the maze between objects
	Distances between objects and gaze points
3. 

LEFT TO DO: 
1. From Polished pickle player, game event stuff
2. From Ultimate Spotlight analysis, the agg data creation.
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook
import matplotlib.animation as animation #RR
import seaborn as sns

import matplotlib as mpl 
mpl.rcParams['animation.ffmpeg_path'] = r'C:/ffmpeg/bin/ffmpeg.exe' ## ffmpeg location

# To fit gamma dists
import scipy.stats as stat
from scipy.special import gammaln
# from scipy.special import gamma ## DID NOT USE THIS, BUT HELPS TO REMEMBER A POINT
from scipy.special import digamma
from scipy.stats import norm
import time


import os

# For locating all objects
from code00_MsPacman_object_locator_ultimate import locate_everything, load_maze
# For calculating distances along the maze
from under_the_hood_things_by_RR.Maze_solver_For_SloPacman import maze_solver_compact
import itertools
from itertools import chain

def show_frame_by_id(i):

	# For efficiency, the image file is read here, only when it's needed. Later on, we would need to use the images to locate objects.
	# Then, I will make the reading more general.
	frame_id = dataset.frame_id.iloc[i]
	# Colors:: blue: 0.5, yellow: 1.0, green: 1.5
	image_file = image_data_subdirectory_for_datafile + "/"+ frame_id +".png"		
	image = plt.imread(image_file)
	plt.imshow(image)
	plt.show()
	plt.close()

def calculate_eucl_distance(x1, x2, y1, y2):
	return ((x1 - x2)**2 + (y1 - y2)**2)**(0.5)

def deal_with_game_file(file_i, new_run_everything_and_save_pickle = 0):
	global dataset, all_unique_colors_pacman
	## Small description of the function:
	# "file_list" contains all file names; so we can just use the index "file_i" to call the one we want. 
	# In the main program, a loop is implemented to call multiple items.
	# Two streams of data: 1. Text files of numerical data on action, gazepoints and etc & 2. image data for game states

## ------------ PART 1 ---------
	## NUMERICAL DATA: Some processing done

	# The following format would be necessary, if the location was relative to the code location as usual.
	# dataset_raw = pd.read_fwf("../ATARIHEADsubset/ms_pacman/f1_action_gaze_data/137_KM_3115947_Dec-12-15-59-59.txt", header = None)

	# BUT! I had redirected the current directory to the action data directory in the main program. 
	# So, we are already at the right place and just the filename inside file_list will do.


	if (new_run_everything_and_save_pickle):

		original_text_file_path = action_data_directory + "/" + file_list[file_i]
		dataset_raw = pd.read_fwf(original_text_file_path, header = None)
		dataset = dataset_raw[0].str.split(',', expand = True, n = 6)

		dataset.columns = dataset.iloc[0].values # grab the first row and set as the header row
		dataset = dataset.iloc[1:].reset_index().drop("index", axis = 1) # take the data less the header row

		dataset = dataset[:-1] # Dropping LAST ROW as it was garbage in this file. Wonder if in all files 
		dataset['gaze_positions'] = dataset['gaze_positions'].apply(lambda x: x.replace("null","0,0") ) 

		dataset['gaze_positions_x'] = np.ones([dataset.shape[0],1])*10
		dataset['gaze_positions_y'] = np.ones([dataset.shape[0],1])*10 

		dataset.gaze_positions = dataset.gaze_positions.apply(lambda x: x.split(",")).apply(lambda x: np.array(x, dtype = float))
		dataset.gaze_positions_x = dataset.gaze_positions.apply(lambda x: x[0::2]) # odd indexed elements are the x-coordinates
		dataset.gaze_positions_y = dataset.gaze_positions.apply(lambda x: x[1::2]) # even indexed elements are the y-coordinates

		dataset[["score", "duration(ms)", "unclipped_reward", "action"]] = dataset[["score", "duration(ms)", "unclipped_reward", "action"]].apply(pd.to_numeric, errors='coerce')


		# Initialize columns for storing (game state) information from images
		list_of_columns_to_create = ["num_pacman", "pacman_point_locx", "pacman_point_locy", "pacman_point_locx_projected", "pacman_point_locy_projected", 
									"num_red_ghost", "red_ghost_point_locx", "red_ghost_point_locy", "red_ghost_point_locx_projected", "red_ghost_point_locy_projected", 
									"num_orange_ghost", "orange_ghost_point_locx", "orange_ghost_point_locy", "orange_ghost_point_locx_projected", "orange_ghost_point_locy_projected", 
									"num_purple_ghost", "purple_ghost_point_locx", "purple_ghost_point_locy", "purple_ghost_point_locx_projected", "purple_ghost_point_locy_projected", 
									"num_cyan_ghost", "cyan_ghost_point_locx", "cyan_ghost_point_locy", "cyan_ghost_point_locx_projected", "cyan_ghost_point_locy_projected", 
									"num_scared_ghosts", "scared_ghosts_point_locx", "scared_ghosts_point_locy", "scared_ghosts_point_locx_projected", "scared_ghosts_point_locy_projected", 
									"num_dots", "dots_point_locx", "dots_point_locy", "dots_point_locx_projected", "dots_point_locy_projected", 
									"num_energy_pills", "energy_pills_point_locx", "energy_pills_point_locy", "energy_pills_point_locx_projected", "energy_pills_point_locy_projected", 
									"fruit_point_locx", "fruit_point_locy", "fruit_point_locx_projected", "fruit_point_locy_projected", 
									"score_OCR", "maze_config_number", "num_life_left"]
		# Add the columns
		for col in list_of_columns_to_create: dataset[col] = np.nan
		dataset["fruit_type_by_color"] = "NA" # different, because we want them as strings


		# Some state information are in the form of arrays. ...
		# ... The columns for these need to be (1) converted to type "object" and ...
		# ... (2) then we will use the "at" function of pandas dataframes for assigning arrays to cells.


		dataset["scared_ghosts_point_locx"] = dataset["scared_ghosts_point_locx"].astype("object")
		dataset["scared_ghosts_point_locy"] = dataset["scared_ghosts_point_locy"].astype("object")
		dataset["scared_ghosts_point_locx_projected"] = dataset["scared_ghosts_point_locx_projected"].astype("object")
		dataset["scared_ghosts_point_locy_projected"] = dataset["scared_ghosts_point_locy_projected"].astype("object")
		dataset["dots_point_locx"] = dataset["dots_point_locx"].astype("object")
		dataset["dots_point_locy"] = dataset["dots_point_locy"].astype("object")
		dataset["dots_point_locx_projected"] = dataset["dots_point_locx_projected"].astype("object")
		dataset["dots_point_locy_projected"] = dataset["dots_point_locy_projected"].astype("object")
		dataset["num_energy_pills"] = dataset["num_energy_pills"].astype("object")
		dataset["energy_pills_point_locx"] = dataset["energy_pills_point_locx"].astype("object")
		dataset["energy_pills_point_locy"] = dataset["energy_pills_point_locy"].astype("object")
		dataset["energy_pills_point_locx_projected"] = dataset["energy_pills_point_locx_projected"].astype("object")
		dataset["energy_pills_point_locy_projected"] = dataset["energy_pills_point_locy_projected"].astype("object")

		# reward ==> score
		# events from score change

	## ------------ PART 1 ---------
		## IMAGE DATA: Path to image files specified here, doing some clever tricks with text file names.
		# Some processing is done in another code. I will later incorporate it here.

		img_binary_maze_by_level, img_binary_pathway_by_level = load_maze()
		# def complete_dataset():
		time_start = time.time()
		for row_i in range(dataset.shape[0]): ## range(450) or ## xx range(dataset.shape[0])
			frame_id = dataset.frame_id.iloc[row_i]
			image_name = image_data_subdirectory_for_datafile + "/"+ frame_id +".png"

			num_pacman, pacman_point_locx, pacman_point_locy, pacman_point_locx_projected, pacman_point_locy_projected, \
			num_red_ghost, red_ghost_point_locx, red_ghost_point_locy, red_ghost_point_locx_projected, red_ghost_point_locy_projected, \
			num_orange_ghost, orange_ghost_point_locx, orange_ghost_point_locy, orange_ghost_point_locx_projected, orange_ghost_point_locy_projected, \
			num_purple_ghost, purple_ghost_point_locx, purple_ghost_point_locy, purple_ghost_point_locx_projected, purple_ghost_point_locy_projected, \
			num_cyan_ghost, cyan_ghost_point_locx, cyan_ghost_point_locy,  cyan_ghost_point_locx_projected, cyan_ghost_point_locy_projected, \
			num_scared_ghosts, scared_ghosts_point_locx, scared_ghosts_point_locy,  scared_ghosts_point_locx_projected, scared_ghosts_point_locy_projected, \
			num_energy_pills, energy_pills_point_locx, energy_pills_point_locy,  energy_pills_point_locx_projected, energy_pills_point_locy_projected, \
			num_dots, dots_point_locx, dots_point_locy,  dots_point_locx_projected, dots_point_locy_projected, \
			fruit_type_by_color, fruit_point_locx, fruit_point_locy, fruit_point_locx_projected, fruit_point_locy_projected, \
			score_OCR, maze_config_number, num_life_left, unique_colors_hex = locate_everything(image_name, img_binary_maze_by_level, img_binary_pathway_by_level)

			# print(pacman_point_locx, pacman_point_locy) #xx
			
			dataset.at[row_i, "num_pacman"] = num_pacman
			dataset.at[row_i, "pacman_point_locx"] = pacman_point_locx
			dataset.at[row_i, "pacman_point_locy"] = pacman_point_locy
			dataset.at[row_i, "pacman_point_locx_projected"] = pacman_point_locx_projected
			dataset.at[row_i, "pacman_point_locy_projected"] = pacman_point_locy_projected
			dataset.at[row_i, "num_red_ghost"] = num_red_ghost
			dataset.at[row_i, "red_ghost_point_locx"] = red_ghost_point_locx
			dataset.at[row_i, "red_ghost_point_locy"] = red_ghost_point_locy
			dataset.at[row_i, "red_ghost_point_locx_projected"] = red_ghost_point_locx_projected
			dataset.at[row_i, "red_ghost_point_locy_projected"] = red_ghost_point_locy_projected
			dataset.at[row_i, "num_orange_ghost"] = num_orange_ghost
			dataset.at[row_i, "orange_ghost_point_locx"] = orange_ghost_point_locx
			dataset.at[row_i, "orange_ghost_point_locy"] = orange_ghost_point_locy
			dataset.at[row_i, "orange_ghost_point_locx_projected"] = orange_ghost_point_locx_projected
			dataset.at[row_i, "orange_ghost_point_locy_projected"] = orange_ghost_point_locy_projected
			dataset.at[row_i, "num_purple_ghost"] = num_purple_ghost
			dataset.at[row_i, "purple_ghost_point_locx"] = purple_ghost_point_locx
			dataset.at[row_i, "purple_ghost_point_locy"] = purple_ghost_point_locy
			dataset.at[row_i, "purple_ghost_point_locx_projected"] = purple_ghost_point_locx_projected
			dataset.at[row_i, "purple_ghost_point_locy_projected"] = purple_ghost_point_locy_projected
			dataset.at[row_i, "num_cyan_ghost"] = num_cyan_ghost
			dataset.at[row_i, "cyan_ghost_point_locx"] = cyan_ghost_point_locx
			dataset.at[row_i, "cyan_ghost_point_locy"] = cyan_ghost_point_locy
			dataset.at[row_i, "cyan_ghost_point_locx_projected"] = cyan_ghost_point_locx_projected
			dataset.at[row_i, "cyan_ghost_point_locy_projected"] = cyan_ghost_point_locy_projected
			dataset.at[row_i, "num_scared_ghosts"] = num_scared_ghosts
			dataset.at[row_i, "scared_ghosts_point_locx"] = scared_ghosts_point_locx
			dataset.at[row_i, "scared_ghosts_point_locy"] = scared_ghosts_point_locy
			dataset.at[row_i, "scared_ghosts_point_locx_projected"] = scared_ghosts_point_locx_projected
			dataset.at[row_i, "scared_ghosts_point_locy_projected"] = scared_ghosts_point_locy_projected
			dataset.at[row_i, "num_dots"] = num_energy_pills
			dataset.at[row_i, "dots_point_locx"] = energy_pills_point_locx
			dataset.at[row_i, "dots_point_locy"] = energy_pills_point_locy
			dataset.at[row_i, "dots_point_locx_projected"] = energy_pills_point_locx_projected
			dataset.at[row_i, "dots_point_locy_projected"] = energy_pills_point_locy_projected
			dataset.at[row_i, "num_energy_pills"] = num_dots
			dataset.at[row_i, "energy_pills_point_locx"] = dots_point_locx
			dataset.at[row_i, "energy_pills_point_locy"] = dots_point_locy
			dataset.at[row_i, "energy_pills_point_locx_projected"] = dots_point_locx_projected
			dataset.at[row_i, "energy_pills_point_locy_projected"] = dots_point_locy_projected
			dataset.at[row_i, "fruit_type_by_color"] = fruit_type_by_color
			dataset.at[row_i, "fruit_point_locx"] = fruit_point_locx
			dataset.at[row_i, "fruit_point_locy"] = fruit_point_locy
			dataset.at[row_i, "fruit_point_locx_projected"] = fruit_point_locx_projected
			dataset.at[row_i, "fruit_point_locy_projected"] = fruit_point_locy_projected
			dataset.at[row_i, "score_OCR"] = score_OCR
			dataset.at[row_i, "maze_config_number"] = maze_config_number
			dataset.at[row_i, "num_life_left"] = num_life_left

			all_unique_colors_pacman = all_unique_colors_pacman + unique_colors_hex

		all_unique_colors_pacman = list(set(all_unique_colors_pacman))
		dataset.to_pickle("updated_datafiles/" + file_list[file_i] [:-4]+"_locate_only_pickle.txt")

	# Some fixes/organizing
	# In point locx and locy values for scared ghosts, Some are ints, some are arrays. Making all arrays here. 
		cols_to_fix_int_to_array = ["scared_ghosts_point_locx", "scared_ghosts_point_locy", "scared_ghosts_point_locx_projected", "scared_ghosts_point_locy_projected"]
		for col_name in cols_to_fix_int_to_array:
			temp = dataset.loc[dataset[col_name].apply(np.shape) == ()][col_name].apply(lambda x: np.array([x]))
			dataset.loc[temp.index, col_name] = temp

		# convert_maze_number to int.
		dataset["maze_config_number"] = dataset.maze_config_number.fillna(-1) # Replace NA with -1. Our code ignores any negative maze numbers, so we are good.
		dataset["maze_config_number"] = dataset.maze_config_number.astype(int)

		# Scores to int
		# First, we replace negative values with NA (for smoothness of score animation) 
		# and then do a forward fill for NAs in the score
		dataset.loc[dataset.score_OCR < 0, "score_OCR"] = np.nan
		dataset["score_OCR"] = dataset.score_OCR.ffill() # Replace NAs with the last non-NA value, "forward fill" in pandas.
		dataset["score_OCR"] = dataset.score_OCR.astype(int)
	# ---------------------- Gaze Data ----------------------------------
# -------------------------------------------------------------------

	# Let's classify fixations and saccades, using std of the points.
	# A main reaon: We want to calculate the distances from the gaze point from the objects, 
	# As the eye data are on particular frames, the gaze points are not really points, rather set of points spread in different ways.
	# We would calculate the distances differently based on how they are spread.

	# During fixations of particular points, they are points spread within a small area. 
	# During saccades or moving gaze one point to another, they are aligned in lines.
	# By default we assume, the gaze points on each frame are all saccades unless it is a fixation.
	# Then, we find and label the fixations on each frame.
	# NOTE: This way, we will identify only the cases with one fixation point per frame.
	# A sad, but necessary simplification for computational efficiency


	# Fixations
		# if the std is below a minimum number of pixels (e.g., pixel_lim_each_of_x_and_y = 4), we classify it as a fixation point on the frame
		xx = [np.std(dataset.gaze_positions_x.iloc[i]) for i in range(dataset.shape[0])]
		yy = [np.std(dataset.gaze_positions_y.iloc[i]) for i in range(dataset.shape[0])]
		xx = np.array(xx); yy = np.array(yy)

		zz = (xx + yy) / 2
		pixel_lim_xy = 4
		pixel_lim_xy_avg = pixel_lim_xy * 0.75

		fixations = np.logical_and(xx < pixel_lim_xy, yy < pixel_lim_xy) # First we take all points that have low stds in each of X and Y directions
		fixations = np.logical_or(fixations, zz <pixel_lim_xy_avg) # Second, we add all points that have low stds on average in both X and Y directions
		no_gaze_points = np.logical_and(xx == 0, yy == 0)

		# dataset["fixation"] = np.array( fixations, dtype = np.int)
		# print(dataset[dataset.fixation ==0].shape)

	# Saccades
		dataset["gaze_type"] = 's' # s= Saccade. 
		dataset.loc[fixations, "gaze_type"] = 'f' # Fixations. 
		dataset.loc[no_gaze_points, "gaze_type"] = 'ng' # No gaze points.

		if (dataset[dataset.gaze_type == 'f'].shape[0] + dataset[dataset.gaze_type == 'ng'].shape[0] + \
			dataset[dataset.gaze_type == 's'].shape[0] == dataset.shape[0]):
			print("YAY! ALL FRAMES ACCOUNTED FOR GAZE CLASSIFICATION!")
		else: 
			print("ALAS! all frames NOT accounted for! %i"% (dataset.shape[0] - dataset[dataset.gaze_type == 'f'].shape[0] - dataset[dataset.gaze_type == 'ng'].shape[0] - \
				dataset[dataset.gaze_type == 's'].shape[0]))

	# Fix mismatches of shape

	# ----------------- GAZE_DATA PROCESSING DONE -----------------------


	# ---------------- NOW, LET'S GET THE DISTANCES BETWEEN THE GAZE POINTS AND THE OBJECTS ---------------------

		# For fixations, euclidean distance between two points -- the fixation point and the object.
		# For saccades, we take the minimum euclidearn distance from the set of points to the objects (a point).
			# as saccades relate more to peripheral vision rather than the fovea, the closest gazepoint to the object shoyld be helpful.

		# ------------------- (1) FIXATIONS ------------------------------------
		indices_of_fixations = dataset[dataset.gaze_type == 'f'].index
		indices_of_saccades = dataset[dataset.gaze_type == 's'].index

		# Now, let's get the point estimates for fixations

		temp_x = dataset.gaze_positions_x[dataset.gaze_type == 'f'].apply(np.mean)
		temp_y = dataset.gaze_positions_y[dataset.gaze_type == 'f'].apply(np.mean)

		dataset["fixation_x"] = np.nan; dataset["fixation_y"] = np.nan
		dataset.loc[dataset.gaze_type == 'f', "fixation_x"] = temp_x
		dataset.loc[dataset.gaze_type == 'f', "fixation_y"] = temp_y

		# Distances over the whole game
		dataset.loc[:,"d_pf"] = calculate_eucl_distance(dataset.pacman_point_locx_projected, dataset.fixation_x, dataset.pacman_point_locy_projected, dataset.fixation_y)
		dataset.loc[:,"d_rgf"] = calculate_eucl_distance(dataset.red_ghost_point_locx_projected, dataset.fixation_x, dataset.red_ghost_point_locy_projected, dataset.fixation_y)
		dataset.loc[:,"d_ogf"] = calculate_eucl_distance(dataset.orange_ghost_point_locx_projected, dataset.fixation_x, dataset.orange_ghost_point_locy_projected, dataset.fixation_y)
		dataset.loc[:,"d_cgf"] = calculate_eucl_distance(dataset.cyan_ghost_point_locx_projected, dataset.fixation_x, dataset.cyan_ghost_point_locy_projected, dataset.fixation_y)
		dataset.loc[:,"d_pgf"] = calculate_eucl_distance(dataset.purple_ghost_point_locx_projected, dataset.fixation_x, dataset.purple_ghost_point_locy_projected, dataset.fixation_y)
		
		# Special attention to fixations on scared ghosts, as multiple in number

		# subdataset_sg_fixations = dataset.loc[((dataset.num_scared_ghosts > 0) & (dataset.gaze_type == "f"))]

		# d_sgf_all = calculate_eucl_distance(subdataset_sg_fixations["scared_ghosts_point_locx_projected"], subdataset_sg_fixations["fixation_x"],  \
		# 		subdataset_sg_fixations["scared_ghosts_point_locy_projected"], subdataset_sg_fixations["fixation_y"])

		# dataset["d_sgf_all"] = d_sgf_all # We store the whole arrays
		# print(d_sgf_all)
		# OLD
		# fx_sg = dataset.loc[((dataset.num_scared_ghosts > 0) & (dataset.gaze_type == "f")), \
		# 					"gaze_positions_x"] # Fixations x coordinates, when scared ghost available

		# fy_sg = dataset.loc[((dataset.num_scared_ghosts > 0) & (dataset.gaze_type == "f")), \
		# 					"gaze_positions_y"] # Fixations y coordinates, when scared ghost available

		# sgx_fixations = dataset.loc[((dataset.num_scared_ghosts > 0) & (dataset.gaze_type == "f")), \
		# 					"scared_ghosts_point_locx_projected"]

		# sgy_fixations = dataset.loc[((dataset.num_scared_ghosts > 0) & (dataset.gaze_type == "f")), \
		# 					"scared_ghosts_point_locy_projected"]


		# sgx_fixations = pd.DataFrame(sgx_fixations.tolist()).values
		# sgy_fixations = pd.DataFrame(sgy_fixations.values.tolist()).values

		# for sg_i in range(4):
		# 	dataset["d_sg%if"%sg_i] = np.nan
		# 	dataset.loc[:, "d_sg%if"%sg_i] = calculate_eucl_distance(fx_sg, sgx_fixations[:, sg_i], fy_sg, sgy_fixations[:, sg_i]).apply(np.min)


		# # dataset.loc[:,"d_sgf"] = calculate_eucl_distance(dataset.scared_ghosts_point_locx_projected, dataset.fixation_x, dataset.scared_ghosts_point_locy_projected, dataset.fixation_y)

		# # xy = dataset.loc[:,"scared_ghosts_point_locx_projected"].isnull()


		# print("mean distance between PACMAN and FIXATION points, in WHOLE GAME: %f"%(np.nanmean(dataset["d_pf"])))
		# print("mean distance between RED GHOST and FIXATION points, in WHOLE GAME: %f"%(np.nanmean(dataset["d_rgf"])))
		# print("mean distance between ORANGE GHOST and FIXATION points, in WHOLE GAME: %f"%(np.nanmean(dataset["d_ogf"])))
		# print("mean distance between CYAN GHOST and FIXATION points, in WHOLE GAME: %f"%(np.nanmean(dataset["d_cgf"])))
		# print("mean distance between PURPLE GHOST and FIXATION points, in WHOLE GAME: %f"%(np.nanmean(dataset["d_pgf"])))
		# # print("mean distance between SCARED GHOST and FIXATION points, in WHOLE GAME: %f"%(np.nanmean(dataset["d_sgf"].apply(np.nanmean) ) ))
		# print("\nScore Achieved%f\n"%max(dataset.score_OCR))

		# ------------------- (2) SACCADES ------------------------------------

		sx = dataset[dataset.gaze_type == "s"].gaze_positions_x
		sy = dataset[dataset.gaze_type == "s"].gaze_positions_y
		px = dataset[dataset.gaze_type == "s"].pacman_point_locx_projected
		py = dataset[dataset.gaze_type == "s"].pacman_point_locy_projected
		rgx = dataset[dataset.gaze_type == "s"].red_ghost_point_locx_projected
		rgy = dataset[dataset.gaze_type == "s"].red_ghost_point_locy_projected
		ogx = dataset[dataset.gaze_type == "s"].orange_ghost_point_locx_projected
		ogy = dataset[dataset.gaze_type == "s"].orange_ghost_point_locy_projected
		cgx = dataset[dataset.gaze_type == "s"].cyan_ghost_point_locx_projected
		cgy = dataset[dataset.gaze_type == "s"].cyan_ghost_point_locy_projected
		pgx = dataset[dataset.gaze_type == "s"].purple_ghost_point_locx_projected
		pgy = dataset[dataset.gaze_type == "s"].purple_ghost_point_locy_projected


		mismatch_index_list = sx[(sx.apply(len) != sy.apply(len))].index

		for i in mismatch_index_list: 
			min_length = min(sx[i].shape[0], sy[i].shape[0])
			sx.loc[i] = sx.loc[i][:min_length]
			sy.loc[i] = sy.loc[i][:min_length]

		dataset.loc[:, "d_ps_min"] = calculate_eucl_distance(sx, px, sy, py).apply(np.min)
		dataset.loc[:, "d_rgs_min"] = calculate_eucl_distance(sx, rgx, sy, rgy).apply(np.min)
		dataset.loc[:, "d_ogs_min"] = calculate_eucl_distance(sx, ogx, sy, ogy).apply(np.min)
		dataset.loc[:, "d_cgs_min"] = calculate_eucl_distance(sx, cgx, sy, cgy).apply(np.min)
		dataset.loc[:, "d_pgs_min"] = calculate_eucl_distance(sx, pgx, sy, pgy).apply(np.min)

		# SPECIAL ATTENTION TO SACCADES ON SCARED GHOSTS!
		time_dist = []
		time_dist.append(time.time())
		subdataset_sg_saccades = dataset.loc[((dataset.num_scared_ghosts > 0) & (dataset.gaze_type == "s"))]

		d_sgs_all = subdataset_sg_saccades.apply(lambda x: [calculate_eucl_distance(x["scared_ghosts_point_locx_projected"][i], x["gaze_positions_x"], \
			x["scared_ghosts_point_locy"][i], x["gaze_positions_y"]) for i in range(int(x["num_scared_ghosts"]))], axis = 1)

		dataset["d_sgs_all"] = d_sgs_all # We store the whole arrays
		time_dist.append(time.time())


		sx_sg = dataset.loc[((dataset.num_scared_ghosts > 0) & (dataset.gaze_type == "s")), \
							"gaze_positions_x"] # Saccades x coordinates, when scared ghost available

		sy_sg = dataset.loc[((dataset.num_scared_ghosts > 0) & (dataset.gaze_type == "s")), \
							"gaze_positions_y"] # Saccades y coordinates

		sgx_saccades = dataset.loc[((dataset.num_scared_ghosts > 0) & (dataset.gaze_type == "s")), \
							"scared_ghosts_point_locx_projected"]

		sgy_saccades = dataset.loc[((dataset.num_scared_ghosts > 0) & (dataset.gaze_type == "s")), \
							"scared_ghosts_point_locy_projected"]

		# Now, we will try to do an all-by-all distance calculation. We do it elegantly,
		# by neatly organizing in np arrays, so that we can use each column separately
		sgx_saccades = pd.DataFrame(sgx_saccades.values.tolist()).values
		sgy_saccades = pd.DataFrame(sgy_saccades.values.tolist()).values

		d_sg_min = np.zeros_like(sgx_saccades)

		for sg_i in range(sgx_saccades.shape[1]):
			# print(sgx_saccades.shape[0])
			dataset["d_sg%is_min"%sg_i] = np.nan
			dataset.loc[:, "d_sg%is_min"%sg_i] = calculate_eucl_distance(sx_sg, sgx_saccades[:, sg_i], sy_sg, sgy_saccades[:, sg_i]).apply(np.min)
		time_dist.append(time.time())
		print(np.diff(time_dist))
	# ---------------- DONE! WE GOT THE DISTANCES BETWEEN THE GAZE POINTS AND THE OBJECTS



	# ---------------------------------------------------------------------------------
	# ---------------- NOW, LET'S GET THE DISTANCES GHOSTS AND PACMAN -----------------
	# ---------------------------------------------------------------------------------

		time_dist = []
		time_dist.append(time.time())
		maze_number_arr = dataset["maze_config_number"] #.tolist()
		frame_id_arr = dataset["frame_id"]

		# REFERENCE: PACMAN
		x1_arr = dataset["pacman_point_locx_projected"]
		y1_arr = dataset["pacman_point_locy_projected"]

		# ------------- (1) red GHOST -----------------
		x2_arr = dataset["red_ghost_point_locx_projected"]
		y2_arr = dataset["red_ghost_point_locy_projected"]
		distances = maze_solver_compact.create_graph_and_return_mazewise_distance_BATCH(maze_number_arr, \
								frame_id_arr, x1_arr, y1_arr, x2_arr, y2_arr)
		distances[distances<= -.1] = np.nan
		dataset["d_rgp"] = distances
		time_dist.append(time.time())
		# ------------- (2) orange GHOST -----------------
		x2_arr = dataset["orange_ghost_point_locx_projected"]
		y2_arr = dataset["orange_ghost_point_locy_projected"]
		distances = maze_solver_compact.create_graph_and_return_mazewise_distance_BATCH(maze_number_arr, \
								frame_id_arr, x1_arr, y1_arr, x2_arr, y2_arr)
		distances[distances<= -.1] = np.nan
		dataset["d_ogp"] = distances

		# ------------- (3) purple GHOST -----------------
		x2_arr = dataset["purple_ghost_point_locx_projected"]
		y2_arr = dataset["purple_ghost_point_locy_projected"]
		distances = maze_solver_compact.create_graph_and_return_mazewise_distance_BATCH(maze_number_arr, \
								frame_id_arr, x1_arr, y1_arr, x2_arr, y2_arr)
		distances[distances<= -.1] = np.nan
		dataset["d_pgp"] = distances

		# ------------- (4) cyan GHOST -----------------
		x2_arr = dataset["cyan_ghost_point_locx_projected"]
		y2_arr = dataset["cyan_ghost_point_locy_projected"]
		distances = maze_solver_compact.create_graph_and_return_mazewise_distance_BATCH(maze_number_arr, \
								frame_id_arr, x1_arr, y1_arr, x2_arr, y2_arr)
		distances[distances<= -.1] = np.nan
		dataset["d_cgp"] = distances
		time_dist.append(time.time())
		# # ------------- (5) scared GHOSTs -----------------
		maze_number_arr_sg = dataset.loc[dataset.num_scared_ghosts > 0, \
							"maze_config_number"]
		frame_id_arr_sg = dataset.loc[dataset.num_scared_ghosts > 0, \
							"frame_id"]
		x1_arr_sg = dataset.loc[dataset.num_scared_ghosts > 0, \
							"pacman_point_locx_projected"]
		y1_arr_sg = dataset.loc[dataset.num_scared_ghosts > 0, \
							"pacman_point_locy_projected"]
		# x2_arr = dataset["scared_ghosts_point_locx_projected"]
		# y2_arr = dataset["scared_ghost_point_locy_projected"]

		sgx = dataset.loc[dataset.num_scared_ghosts > 0, \
							"scared_ghosts_point_locx_projected"]

		sgy = dataset.loc[dataset.num_scared_ghosts > 0, \
							"scared_ghosts_point_locy_projected"]

		# Now, we will try to do an all-by-all distance calculation. We do it elegantly,
		# by neatly organizing in np arrays, so that we can use each column separately
		sgx = pd.DataFrame(sgx.values.tolist()).values
		sgy = pd.DataFrame(sgy.values.tolist()).values
		time_dist.append(time.time())

		for sg_i in range(4):
			dataset["d_sg%ip"%sg_i] = np.nan
			x2_arr_sg = sgx[:, sg_i]
			y2_arr_sg = sgy[:, sg_i]
			
			print(sg_i, maze_number_arr_sg.shape, frame_id_arr_sg.shape, x1_arr_sg.shape, y1_arr_sg.shape, x2_arr_sg.shape, y2_arr_sg.shape)
			dataset.loc[dataset.num_scared_ghosts > 0, "d_sg%ip"%sg_i] = maze_solver_compact.create_graph_and_return_mazewise_distance_BATCH(maze_number_arr_sg, \
								frame_id_arr_sg, x1_arr_sg, y1_arr_sg, x2_arr_sg, y2_arr_sg)

		time_dist.append(time.time())
		print(np.diff(time_dist))
	#-----------------------------------------------------------------------------

		pickle_filename = "updated_datafiles/"+file_list[file_i] [:-4]+"_ultimate_pickle.txt"
		dataset.to_pickle(pickle_filename)
	else:
		# dataset = pd.read_pickle("updated_datafiles/" + file_list[file_i] [:-4]+"_locate_only_pickle.txt")
		dataset = pd.read_pickle("updated_datafiles/"+file_list[file_i] [:-4]+"_ultimate_pickle.txt")
	# print("Process_time = %f" %(time.time() - time_old))

	return(dataset)


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

## -------------------------- Animation Functions -------------------------
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


# NOTE: About the below function
# The first argument will be implcit in "fargs" later.
	# It will come from the frames parameter (i.e., range(num_frames) in our case) in the function.
# We can add as many arguments we want! So simple. More ax, more data etc.

# def animate_gameplay_experimental_simple(i_frame, ax1, ax2):
# 	ax1.clear()
# 	ax2.clear()

# 	frame_id = dataset.frame_id.iloc[i_frame]
# 	# Colors:: blue: 0.5, yellow: 1.0, green: 1.5
# 	image_file = image_data_subdirectory_for_datafile + "/"+ frame_id +".png"		
# 	image = plt.imread(image_file)
# 	# print(image_file)

# 	ax1.imshow(image)
	# ax2.imshow(image)



# ---------------- OLD, All about Animation: START ------------------------------------


# # init may be useful in some cases. I avoided it as I didn't get and need it.

# def init():
#     plot21.set_data(aa)
#     plot12.set_data(10,10)
#     return plot11, plot12,

def animate_gameplay_update(i, plot1_background, plot1_pacman, plot1_red_ghost, plot1_orange_ghost, plot1_purple_ghost, plot1_cyan_ghost, plot1_scared_ghosts, plot1_gaze,\
			plot2_background, plot2_dots, plot2_energy_pills, plot2_fruit, plot2_pacman, plot2_red_ghost, plot2_orange_ghost, plot2_purple_ghost, plot2_cyan_ghost, plot2_scared_ghosts, \
			plot2_score_OCR, plot2_num_life_left, plot2_fruit_info, plot2_gaze,):

# ------------ 1st Animation, on the left ------------------------------
	frame_id = dataset.frame_id.iloc[i]
	image_file = image_data_subdirectory_for_datafile + "/"+ frame_id +".png"		
	image = plt.imread(image_file)

	plot1_background.set_data(image)
	if (dataset.num_pacman[i] != 0): plot1_pacman.set_data([dataset.pacman_point_locy_projected[i]], [dataset.pacman_point_locx_projected[i]])
	if (dataset.num_red_ghost[i] != 0): plot1_red_ghost.set_data([dataset.red_ghost_point_locy_projected[i]], [dataset.red_ghost_point_locx_projected[i]])
	if (dataset.num_orange_ghost[i] != 0): plot1_orange_ghost.set_data([dataset.orange_ghost_point_locy_projected[i]], [dataset.orange_ghost_point_locx_projected[i]])
	if (dataset.num_purple_ghost[i] != 0): plot1_purple_ghost.set_data([dataset.purple_ghost_point_locy_projected[i]], [dataset.purple_ghost_point_locx_projected[i]])
	if (dataset.num_cyan_ghost[i] != 0): plot1_cyan_ghost.set_data([dataset.cyan_ghost_point_locy_projected[i]], [dataset.cyan_ghost_point_locx_projected[i]])
	if (dataset.num_scared_ghosts[i] != 0): plot1_scared_ghosts.set_data([dataset.scared_ghosts_point_locy_projected[i]], [dataset.scared_ghosts_point_locx_projected[i]])
	plot1_gaze.set_data(dataset.gaze_positions_x.iloc[i], dataset.gaze_positions_y.iloc[i])

# ------------ 1st Animation, on the left ------------------------------

	# TEMPORARY LINES START -- TO DELETE after rerun
	dataset["maze_config_number"] = dataset.maze_config_number.fillna(-1)
	dataset["maze_config_number"] = dataset.maze_config_number.astype(int)
	dataset.loc[dataset.score_OCR < 0, "score_OCR"] = np.nan
	dataset["score_OCR"] = dataset.score_OCR.ffill() # Replace NAs with the last non-NA value, "forward fill" in pandas.
	dataset["score_OCR"] = dataset.score_OCR.astype(int)
	# TEMPORARY LINES END

	maze_config_number = dataset.maze_config_number.iloc[i]
	maze_to_plot = img_maze[maze_config_number]
	plot2_background.set_data(maze_to_plot)
	plot2_dots.set_data([dataset.dots_point_locy[i]], [dataset.dots_point_locx[i]])
	plot2_energy_pills.set_data([dataset.energy_pills_point_locy[i]], [dataset.energy_pills_point_locx[i]])
	plot2_fruit.set_data([dataset.fruit_point_locy[i]], [dataset.fruit_point_locx[i]])
	try: plot2_fruit.set_color(dataset.fruit_type_by_color.iloc[i])
	except KeyError: print("KeyError with color: " + dataset.fruit_type_by_color.iloc[i])
	if (dataset.num_pacman[i] != 0): plot2_pacman.set_data([dataset.pacman_point_locy_projected[i]], [dataset.pacman_point_locx_projected[i]])
	if (dataset.num_red_ghost[i] != 0): plot2_red_ghost.set_data([dataset.red_ghost_point_locy_projected[i]], [dataset.red_ghost_point_locx_projected[i]])
	if (dataset.num_orange_ghost[i] != 0): plot2_orange_ghost.set_data([dataset.orange_ghost_point_locy_projected[i]], [dataset.orange_ghost_point_locx_projected[i]])
	if (dataset.num_purple_ghost[i] != 0): plot2_purple_ghost.set_data([dataset.purple_ghost_point_locy_projected[i]], [dataset.purple_ghost_point_locx_projected[i]])
	if (dataset.num_cyan_ghost[i] != 0): plot2_cyan_ghost.set_data([dataset.cyan_ghost_point_locy_projected[i]], [dataset.cyan_ghost_point_locx_projected[i]])
	if (dataset.num_scared_ghosts[i] != 0): plot2_scared_ghosts.set_data([dataset.scared_ghosts_point_locy_projected[i]], [dataset.scared_ghosts_point_locx_projected[i]])
	plot2_score_OCR.set_text(dataset.score_OCR[i])
	plot2_num_life_left.set_text(r"Lives left = " + str(int(dataset.num_life_left.iloc[i])))
	# color_to_fruit_dict = {"#b83232":"Cherry/Strawberry", "#a2a22a":"Pretzel", "#c66c3a":"Orange"}
	# plot2_fruit_info.set_data([130], [175])
	try: plot2_fruit_info.set_color(dataset.fruit_type_by_color.iloc[i])
	except KeyError: print("KeyError with color: " + dataset.fruit_type_by_color.iloc[i])
	plot2_gaze.set_data(dataset.gaze_positions_x.iloc[i], dataset.gaze_positions_y.iloc[i])

	return plot1_background, plot1_pacman, plot1_red_ghost, plot1_orange_ghost, plot1_purple_ghost, plot1_cyan_ghost, plot1_scared_ghosts, plot1_gaze,\
			plot2_background, plot2_dots, plot2_energy_pills, plot2_fruit, plot2_pacman, plot2_red_ghost, plot2_orange_ghost, plot2_purple_ghost, plot2_cyan_ghost, plot2_scared_ghosts, \
			plot2_score_OCR, plot2_num_life_left, plot2_fruit_info, plot2_gaze,
	# NOTE and interesting point: The return order determines the layers!


def animate_gameplay_master(dataset, to_save_animation = 0):
	# ------------- ANIMATION, OLD but working great------------------------
	# Note: I finally got rid of the warnings about set_data.
	# Turns out, all I needed to do is make everything a "sequence" in the animation function. 
	# Like here: plot1_pacman.set_data([dataset.pacman_point_locy_projected[i]], [dataset.pacman_point_locx_projected[i]])
									# See, how the single integers need to be in a "bucket" or a list.
	
	interval_span = 10 # Change duration of each frame here, in miliseconds
	num_frames = dataset.shape[0] # Specify how many frames you wanna play
	range_of_frames = np.arange(0, num_frames)

	fig = plt.figure(figsize = (10, 6))
	ax1 = fig.add_subplot(121, aspect = 'equal', autoscale_on = True)
	ax2 = fig.add_subplot(122, sharey = ax1)

	random_image_initialization = np.random.random([210,160,3])
	plot1_background = ax1.imshow(random_image_initialization)
	# plot13, = ax1.plot(1,1,'g.') # Empty layer for object locations ("green .")
	plot1_pacman, = ax1.plot(0,0, marker = "$P$", color = 'w')
	plot1_red_ghost, = ax1.plot(0,0, marker = "$RG$", color = 'w')
	plot1_orange_ghost, = ax1.plot(0,0, marker = "$OG$", color = 'w')
	plot1_purple_ghost, = ax1.plot(0,0, marker = "$PG$", color = 'w')
	plot1_cyan_ghost, = ax1.plot(0,0, marker = "$CG$", color = 'w')
	plot1_scared_ghosts, = ax2.plot([0, 1, 2], [0, 1, 2], linestyle = "", marker = "$SG$", color = 'w')
	plot1_gaze, = ax1.plot(0,0,'wx') # Empty layer for gazepoints ("white x")
	ax1.grid(); ax1.axis('off')

	plot2_background = ax2.imshow(random_image_initialization, cmap = "gray") # gray, Purples_r, Blues, spring
	plot2_dots, = ax2.plot([0, 1, 2], [0, 1, 2], linestyle = "", marker = "8", markersize = 2, color = 'hotpink')
	plot2_energy_pills, = ax2.plot([0, 1, 2], [0, 1, 2], linestyle = "", marker = "D", color = 'darkorchid')
	plot2_fruit, = ax2.plot(0,0, marker = "*", color = 'red', markeredgecolor = "black", markersize = 12)
	# plot23, = ax2.plot(1,1,'g.') # Empty layer for object locations ("green .")
	plot2_pacman, = ax2.plot(0,0, marker = "o", color = 'yellow', markeredgecolor = "black", markersize = 12)
	plot2_red_ghost, = ax2.plot(0,0, marker = "^", color = 'red', markeredgecolor = "black", markersize = 12)
	plot2_orange_ghost, = ax2.plot(0,0, marker = "^", color = 'orange', markeredgecolor = "black", markersize = 12)
	plot2_purple_ghost, = ax2.plot(0,0, marker = "^", color = 'purple', markeredgecolor = "black", markersize = 12)
	plot2_cyan_ghost, = ax2.plot(0,0, marker = "^", color = 'cyan', markeredgecolor = "black", markersize = 12)
	plot2_scared_ghosts, = ax2.plot([0, 1, 2], [0, 1, 2], linestyle = "", marker = "^", color = 'blue', markeredgecolor = "black", markersize = 12)
	plot2_gaze, = ax2.plot(0,0,'x', color = "limegreen") # Empty layer for gazepoints ("white x")

	props = dict(boxstyle='round', facecolor='pink', alpha=0.5)		
	plot2_score_OCR = ax2.text(x = 100, y = 190, s = r"0", fontsize = 14, color = "mediumspringgreen", \
		horizontalalignment = "right", verticalalignment = "center", bbox = props)
	plot2_num_life_left = ax2.text(x = 5, y = 180, s = r"", fontsize = 10, color = "yellow", \
		horizontalalignment = "left", verticalalignment = "bottom", bbox = props)
	plot2_fruit_info, = ax2.plot([130], [175], marker = "*", markersize = 12, color = "black", markeredgecolor = "w")

	ax2.grid(); ax2.axis('off')	


	ani_pacman = animation.FuncAnimation(fig, animate_gameplay_update, range_of_frames,
                          interval= interval_span, blit = True, repeat=False, 
                          fargs = (plot1_background, plot1_pacman, plot1_red_ghost, plot1_orange_ghost, plot1_purple_ghost, plot1_cyan_ghost, plot1_scared_ghosts, plot1_gaze,\
			plot2_background, plot2_dots, plot2_energy_pills, plot2_fruit, plot2_pacman, plot2_red_ghost, plot2_orange_ghost, plot2_purple_ghost, plot2_cyan_ghost, plot2_scared_ghosts, \
			plot2_score_OCR, plot2_num_life_left, plot2_fruit_info, plot2_gaze,))
	
# create a new folder to save all text and image data together


	if to_save_animation:
		anim_file_id = file_list[file_i][:-20]
		directory_for_saving_animations = "gameplay_animations"
		if not os.path.exists(directory_for_saving_animations): os.makedirs(directory_for_saving_animations)
		f = directory_for_saving_animations + "/anim_pacman_session_%i_file_%s.mp4"%(file_i+1, anim_file_id)
		print(f)
		time_old = time.time()
		writervideo = animation.FFMpegWriter(fps=20) 
		ani_pacman.save(f, writer=writervideo)
		print("time_taken to save video: %.2f seconds" %(time.time() - time_old))
	
	else: plt.show()
	plt.close()




# ---------------- PART 3: END ----------------------


if __name__ == "__main__":
	# print("xx")

	action_data_directory = "ms_pacman"
	image_data_directory = "extracted_datafiles"
	file_list = os.listdir(action_data_directory)
	# We want only the text files.
	## The following gives a list of all files with "*.xyz" extention.
	file_list = [elem for elem in file_list if elem.endswith(".txt")]
	print("Number of session files:", len(file_list))
	all_unique_colors_pacman = []

	## ------------------ LOAD AND PREP MAZE IMAGES -------------------------------------
	# We will need them for our animations
	img_maze = load_maze()[0]
	# We remove the scoring area and borders, and just keep the playing area
	maze_filter = np.ones_like(img_maze[0])
	maze_filter[:1,:] = 0 # Top border removed
	maze_filter[171:,:] = 0 # Scoring at the bottom removed
	maze_filter[65:94, 65:94] = 0 # Omit home of the ghosts too

	for i in range(len(img_maze)): img_maze[i] = img_maze[i] * maze_filter

# -------------------------------------------------------------------------------
# ------------------ DEAL WITH GAME FILE AND ANIMATE ----------------------------
# -------------------------------------------------------------------------------
	dataset_all = []
	for file_i in np.arange(6, len(file_list)):
		image_data_subdirectory_for_datafile = image_data_directory + "/" + file_list[file_i] [:-4]
		print("Current file: (i=%i)"%file_i, file_list[file_i])
		# exit()
		dataset = deal_with_game_file(file_i = file_i,  new_run_everything_and_save_pickle = 1)
		dataset_all.append(dataset)

	for file_i in np.arange(len(file_list)):
		dataset = dataset_all[file_i]
		animate_gameplay_master(dataset, to_save_animation = 1)


	# --------- NEW Experiment with Animation ---------------------------
	# UPDATE: So simple to write, but so slow to run.
	# Main problem: Could not get the blit (which removes previous data) working, so became really memory consuming. And SLOOOOOW...

		# fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 10))

		# To have a constant thing in the background, we can draw here.
		# plt.suptitle("Changes in Networks of Tuning Performance with Expertise")

		# ani = animation.FuncAnimation(fig, animate_gameplay_new, frames = dataset.shape[0],
		# 							   fargs = (ax1, ax2),
		# 							   interval = 10, blit = True, repeat = False)

		# plt.show()
		# plt.close()




	## Maze match not found 
	# i = 0: 2 [(8801, 8901), (17501, 17601)]
	# i = 1: 2 [(8201, 8301), (15301, 15401)]
	# i = 2: 2 [(7601, 7701), (15901, 16001)]
	# i = 3: 2 [(8201, 8301), (15401, 15501)]
	# i = 4: 2 [(8501, 8601), (15901, 16001)]
	# i = 5 to 9: 0 []