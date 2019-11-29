from project_data import link_base_image, link_base_image_large_annotated, link_base_image_warning, ucl_east_image
from project_data import shape_y, shape_x, thickness_lines, root_participation_directory
from project_data import reference_directory_images, reference_directory, overall_results_directory
from project_data import link_outcome_failure, link_outcome_success, node_coords, threshold_distance
from project_data import node_coords_large, color_canvas_rgb, node_coords_detailed
from project_data import site_scale_factor, massing_height
from project_data import ucl_east_development_area, ucl_east_student_population, ucl_east_research_area
from project_data import ratio_accomodation_base, ratio_accomodation_plinth, ratio_accomodation_tower, m2accomodation_per_student
from project_data import ratio_research_base, ratio_research_plinth, ratio_research_tower, databse_filepath
#import os
import numpy as np
import cv2
import os
import keras
import re
import scipy
import skimage

from tinydb import TinyDB, Query
import tinydb

# ------------------------------------------------------------------------------------
# Imports locally defined functions
# ------------------------------------------------------------------------------------  
from imagenet_utils import preprocess_input
from graph_form_image import path_graph
from overall_analysis import basic_line_drawing, bundle_drawing
from database_management import data_to_database
from basic_drawing_functions import site_area, pts_to_polylines, draw_paths, draw_paths_base, draw_base_large

# ------------------------------------------------------------------------------------
# Imports locally defined functions
# ------------------------------------------------------------------------------------  
from drawing_app_functions import generate_image, pts_to_polylines, draw_paths_base, draw_land_use_analysis, report_land_use, drawscapes_draw_base, drawscapes_feedback_function
from imagenet_utils import preprocess_input
from graph_form_image import path_graph
from overall_analysis import basic_line_drawing, bundle_drawing
from tSNE import plot_tsne


    
import tSNE 

#%%
# ------------------------------------------------------------------------------------
# Generates conclussion drawings for all categories (lines, massing and land uses)
# ------------------------------------------------------------------------------------  

session_user =  '1574982838322'
#millis = 1574964494624

root_data = root_participation_directory
session_folder=os.path.join(root_data, session_user)
folder_name = session_user
#file_name= session_user + '_' + str(millis)


filepath_np_massing = os.path.join(session_folder, session_user + '_lines.npy')
filepath_np_massing_type = os.path.join(session_folder, session_user + '_lines_type.npy')

ptexport=np.load(filepath_np_massing).astype(int)
points=ptexport.tolist()
print (points)
line_type=np.load(filepath_np_massing_type).astype(int)
#%%
declared_style = 4
task = 2

#generate_image(ptexport, line_type, session_folder, file_name, folder_name, declared_style, task)


#%%



print (line_type)
polylines  = pts_to_polylines(points, line_type)[0]
linetype = pts_to_polylines (points, line_type) [1]
#

#land_uses = draw_land_use_analysis (polylines, line_type, session_folder, session_user)
#print(land_uses)
#%%
#data_to_database (polylines, linetype, folder_name, exercise = 'lines', extract_features = 'True')

#generates data

style_save = [4]
data=[]
data.append(style_save)
data.append(points[0])
data.append(points[1])
data.append(points[2])
data.append(line_type.tolist())
print (data)
file_name = session_user
task=2
drawscapes_feedback_function (data, file_name, session_folder, folder_name,task)


#%%

db = TinyDB(databse_filepath)

