
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
from project_data import shape_y, shape_x, thickness_lines
from project_data import reference_directory_images, reference_directory, overall_results_directory
from project_data import link_outcome_failure, link_outcome_success, node_coords, threshold_distance
from project_data import node_coords_large, color_canvas_rgb, node_coords_detailed
from project_data import site_scale_factor, massing_height
from project_data import ucl_east_development_area, ucl_east_student_population, ucl_east_research_area

from drawing_app_functions import generate_image
# ------------------------------------------------------------------------------------
# Imports locally defined functions
# ------------------------------------------------------------------------------------  
from imagenet_utils import preprocess_input
from graph_form_image import path_graph
from overall_analysis import basic_line_drawing, bundle_drawing

#%%


session_user =  '1574168668181'
millis = 1574173840457
file_name = session_user
root_data = 'C:\\Users\\ucbqeri\\Documents\\GitHub\\Flask_Blog\\Drawscapes_2\\data'
session_folder=os.path.join(root_data, session_user)
folder_name = session_user

#filepath_np = os.path.join(se$ssion_folder, str(file_name) + '.npy')
filepath_np_lines = os.path.join(session_folder, str(file_name) + '_lines.npy')
filepath_np_lines_type = os.path.join(session_folder, str(file_name) + '_lines_type.npy')

#file=np.load(filepath_np).tolist()
file_lines=np.load(filepath_np_lines)
file_lines_type=np.load(filepath_np_lines_type).tolist()
print(file_lines)
print(file_lines_type)
ptexport = file_lines
line_type = file_lines_type


declared_style = 4
task = 2

generate_image (ptexport, line_type, session_folder, file_name, folder_name, declared_style, task)
