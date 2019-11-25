
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

#session_user =  '1574548541279'
#millis = 1574548557738
#
#root_data = root_participation_directory
#session_folder=os.path.join(root_data, session_user)
#folder_name = session_user
#file_name= session_user + '_' + str(millis)
#
#
#filepath_np_massing = os.path.join(session_folder, session_user + '_massing.npy')
#filepath_np_massing_type = os.path.join(session_folder, session_user + '_massing_type.npy')
#
#ptexport=np.load(filepath_np_massing).astype(int)
#points=ptexport.tolist()
#print (points)
#line_type=np.load(filepath_np_massing_type).astype(int)
#print (line_type)
#polylines  = pts_to_polylines(points, line_type)[0]
#linetype = pts_to_polylines (points, line_type) [1]
##
#
#land_uses = draw_land_use_analysis (polylines, line_type, session_folder, session_user)
#print(land_uses)

#%%

#generates data
#style_save = [4]
#data=[]
#data.append(style_save)
#data.append(points[0])
#data.append(points[1])
#data.append(points[2])
#data.append(line_type.tolist())
#print (data)
#file_name = session_user
#report_land_use (data, file_name, session_folder, folder_name)


#%%

sketches=os.listdir(overall_results_directory)
#----------------------------------------------------------------------------------------
# Generate lists of np data of sketches in directory
#----------------------------------------------------------------------------------------
def list_line_dir():
    list_files = [] # file names
    list_lines=[] # np with lines
    list_lines_type=[] # np wth types of line
    list_ln = [] #np with simplified lines
    
    #generates list of files
    for i in sketches:          
        if not i == 'Thumbs.db':
            if re.findall('_lines.npy',i) :
                list_lines.append(i)
                list_files.append(i.replace('_lines', ''))
                list_lines_type.append(i.replace('_lines', '_lines_type'))
                list_ln.append(i.replace('_lines', '_ln'))
    
    return list_files, list_lines, list_lines_type, list_ln

#----------------------------------------------------------------------------------------
# removes items with length = 0, generates images if not there and  re-builds deffinitieve list
#----------------------------------------------------------------------------------------
def clean_line_list_dir():
    list_directory = list_line_dir()
    list_files =    list_directory[0]
    list_lines = list_directory[1]
    list_lines_type = list_directory[2]
    list_ln = list_directory[3]
    for i in range (0, len(list_files)):
        filepath_np_lines = os.path.join(overall_results_directory, list_lines[i])
        ptexport=np.load(filepath_np_lines).astype(int)
        if len(ptexport[0]) == 0: #goes through list and removes those with length = 0 
            os.remove(os.path.join(overall_results_directory,list_lines[i]))
            os.remove(os.path.join(overall_results_directory,list_lines_type[i])) 
            if os.path.exists(os.path.join(overall_results_directory,list_ln[i])):
                os.remove(os.path.join(overall_results_directory,list_ln[i]))
        else:
            image_file = list_lines[i].replace('.npy', '.jpg') #goes through list and generates the image if missing
            if os.path.exists(os.path.join(overall_results_directory,image_file)) == False:
                filepath_np_lines_type = os.path.join(overall_results_directory, list_lines_type[i])
                line_type=np.load(filepath_np_lines_type).astype(int)
                points=ptexport.tolist()
                polylines  = pts_to_polylines(points, line_type)[0]
                image_file = image_file.replace('.jpg' , '')
                draw_paths (polylines, line_type, overall_results_directory,image_file)
    return list_line_dir()


list_directory = clean_line_list_dir()        

print(list_directory[0])




#%%


#def tsne_generation(item):
item='lines'

list_directory = clean_line_list_dir()    
print("Loading VGG19 pre-trained model...")
base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)

imgs, X = [], []
for sketch in list_directory[1]:
    sketch = sketch.replace('npy','jpg')
    filename_full = os.path.join(overall_results_directory,sketch)
    img = image.load_img(filename_full, target_size=(224, 224))  # load image
    imgs.append(np.array(img))  # Adds it into the overall list

    # Pre-process for model input
    img = image.img_to_array(img)  # convert to array
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    print('extracting features for' + sketch)
    features = model.predict(img).flatten()  # features
    X.append(features)  # append feature extractor

X = np.array(X)  # feature vectors
imgs = np.array(imgs)  # images
print("imgs.shape = {}".format(imgs.shape))
print("X_features.shape = {}\n".format(X.shape))


#%%
    
tsne_filename = os.path.join(overall_results_directory, 'tsne_'+item+'.jpg')
print("Plotting tSNE to {}...".format(tsne_filename))
zoom = 0.3
plot_tsne(imgs, X, tsne_filename)



#tsne_generation('lines')

#imgs = []
#X= []
#
#
#
#        
#            
#for i in range (0, len(list_files)):
#    #reads the np data ansd calls developemnt of drawing
#    filepath_np_lines = os.path.join(overall_results_directory, list_lines[i])
#    
#
#    ptexport=np.load(filepath_np_lines).astype(int)
##    print(list_files[i])
#    print(str(list_files[i])  + '-' +  str(len(ptexport[0])))
#    points=ptexport.tolist()
#    line_type=np.load(filepath_np_lines_type).astype(int)
#    polylines  = pts_to_polylines(points, line_type)[0]
#    linetype = pts_to_polylines (points, line_type) [1]
#    
#    sketch_image = draw_paths (polylines, line_type, overall_results_directory,list_lines[i])



#plt.imshow(sketch_image)

