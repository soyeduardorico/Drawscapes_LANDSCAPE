
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

from drawing_app_functions import generate_image, pts_to_polylines, draw_paths_base
# ------------------------------------------------------------------------------------
# Imports locally defined functions
# ------------------------------------------------------------------------------------  
from imagenet_utils import preprocess_input
from graph_form_image import path_graph
from overall_analysis import basic_line_drawing, bundle_drawing

#%%


# ------------------------------------------------------------------------------------
# Generates conclussion drawings for all categories (lines, massing and land uses)
# ------------------------------------------------------------------------------------  
def generate_all_drawings (session_user):
    root_data = root_participation_directory
    session_folder=os.path.join(root_data, session_user)
    extension_names =  ['_lines', '_massing', '_land_uses']
    imgtotal=cv2.imread(link_base_image) # this will be 
    for i in extension_names: # iterates thorugh categories
        file_name = session_user + i 
    
        filepath_np_lines = os.path.join(session_folder, str(file_name) + '.npy') # loads lines
        filepath_np_lines_type = os.path.join(session_folder, str(file_name) + '_type.npy') #loads line types
            
        ptexport=np.load(filepath_np_lines).astype(int)
        points=ptexport.tolist()
        line_type=np.load(filepath_np_lines_type).astype(int)
        pols  = pts_to_polylines(points, line_type)[0]
        linetype = pts_to_polylines (points, line_type) [1]
    
        img=cv2.imread(link_base_image)
        
        imgtotal = draw_paths_base (pols, linetype, session_folder, file_name, imgtotal)
        
        draw_paths_base (pols, linetype, session_folder, file_name, img) # calls drawing function
    
    file_name = os.path.join(session_folder, session_user + '_combined.jpg')
    cv2.imwrite(file_name,imgtotal)

session_user =  '1574371816327'
generate_all_drawings (session_user)


##%%
#print(file_lines)
#print(file_lines_type)
#ptexport = file_lines
#line_type = file_lines_type
#
#
#declared_style = 4
#task = 2
#
#generate_image (ptexport, line_type, session_folder, file_name, folder_name, declared_style, task)
#filepath_np = os.path.join(session_folder, folder_name + '_lines.npy')
#ptexport2 = np.load(filepath_np).astype(int)  
#ptexport =  np.concatenate((ptexport,ptexport2),axis=1) # appends lines already drawn for paths
#points=ptexport.tolist()
#pols  = pts_to_polylines(points, line_type)[0]
#linetype = pts_to_polylines (points, line_type) [1]
#draw_paths_base (pols, linetype, session_folder, file_name)