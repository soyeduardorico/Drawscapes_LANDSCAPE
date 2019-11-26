
import numpy as np

import os
#import os
import numpy as np
import cv2
import os
import keras
import re
import scipy
import skimage

from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg19 import VGG19
from sklearn.externals import joblib
from skimage import morphology
from skimage import color
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from matplotlib import pyplot as plt #used for debuggin purposes

# ------------------------------------------------------------------------------------
# Imports project variables
# ------------------------------------------------------------------------------------  
from project_data import link_base_image, link_base_image_large_annotated, link_base_image_warning, ucl_east_image
from project_data import shape_y, shape_x, thickness_lines, root_participation_directory
from project_data import reference_directory_images, reference_directory, overall_results_directory
from project_data import link_outcome_failure, link_outcome_success, node_coords, threshold_distance
from project_data import node_coords_large, color_canvas_rgb, node_coords_detailed
from project_data import site_scale_factor, massing_height
from project_data import ucl_east_development_area, ucl_east_student_population, ucl_east_research_area

# ------------------------------------------------------------------------------------
# Imports locally defined functions
# ------------------------------------------------------------------------------------  
from drawing_app_functions import draw_paths, generate_image, pts_to_polylines, draw_paths_base, draw_land_use_analysis, report_land_use
from imagenet_utils import preprocess_input
from graph_form_image import path_graph
from overall_analysis import basic_line_drawing, bundle_drawing
from tSNE import plot_tsne

#%%
# ------------------------------------------------------------------------------------
# Generates conclussion drawings for all categories (lines, massing and land uses)
# ------------------------------------------------------------------------------------  

session_user =  '1574548541279'
millis = 1574548557738

root_data = root_participation_directory
session_folder=os.path.join(root_data, session_user)
folder_name = session_user
file_name= session_user + '_' + str(millis)


filepath_np_massing = os.path.join(session_folder, session_user + '_massing.npy')
filepath_np_massing_type = os.path.join(session_folder, session_user + '_massing_type.npy')

ptexport=np.load(filepath_np_massing).astype(int)
points=ptexport.tolist()
print (points)
line_type=np.load(filepath_np_massing_type).astype(int)
print (line_type)
polylines  = pts_to_polylines(points, line_type)[0]
linetype = pts_to_polylines (points, line_type) [1]
#

land_uses = draw_land_use_analysis (polylines, line_type, session_folder, session_user)
print(land_uses)

#%%

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
report_land_use (data, file_name, session_folder, folder_name)


#%%

