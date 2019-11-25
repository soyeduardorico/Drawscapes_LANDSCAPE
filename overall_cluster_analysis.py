
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

from sklearn import manifold, datasets

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
    list_massing=[]
    list_massing_type = []
    
    #generates list of files
    for i in sketches:          
        if not i == 'Thumbs.db':
            if re.findall('_lines.npy',i) :
                list_lines.append(i)
                list_files.append(i.replace('_lines', ''))
                list_lines_type.append(i.replace('_lines', '_lines_type'))
                list_ln.append(i.replace('_lines', '_ln'))
            if re.findall('_massing.npy',i):
                list_massing.append(i)
                list_massing_type.append(i.replace('_massing', '_massing_type'))
    
    return list_files, list_lines, list_lines_type, list_ln, list_massing, list_massing_type

#----------------------------------------------------------------------------------------
# removes items with length = 0, generates images if not there and  re-builds deffinitieve list
#----------------------------------------------------------------------------------------
def clean_line_list_dir():
    list_directory = list_line_dir()
    list_files =    list_directory[0]
    list_lines = list_directory[1]
    list_lines_type = list_directory[2]
    list_ln = list_directory[3]
    list_massing = list_directory[4]
    list_massing_type =list_directory[5]
    #first clean for line drawings
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
    
    #second clean for massing drawings
    for i in range (0,len(list_massing)):
        filepath_np_massing = os.path.join(overall_results_directory, list_massing[i])
        ptexport=np.load(filepath_np_massing).astype(int)        
        if len(ptexport[0]) == 0: #goes through list and removes those with length = 0 
            os.remove(os.path.join(overall_results_directory,list_massing[i]))
            os.remove(os.path.join(overall_results_directory,list_massing_type[i]))
        else:
            image_file = list_massing[i].replace('.npy', '.jpg') #goes through list and generates the image if missing
            if os.path.exists(os.path.join(overall_results_directory,image_file)) == False:
                filepath_np_massing_type = os.path.join(overall_results_directory, list_massing_type[i])
                massing_type=np.load(filepath_np_massing_type).astype(int)
                points=ptexport.tolist()
                polylines  = pts_to_polylines(points, massing_type)[0]
                image_file = image_file.replace('.jpg' , '')
                draw_paths (polylines, massing_type, overall_results_directory,image_file)
                
    return list_line_dir()

list_directory = clean_line_list_dir()        


#%%


#def tsne_generation(item):
item='lines'

#loads model
list_directory = clean_line_list_dir()    
print("Loading VGG19 pre-trained model...")
base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)

#develops features form images
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

# appends the featueres and images into large file
X = np.array(X)  # feature vectors
imgs = np.array(imgs)  # images



#%%

item='lines'

#develps Tsne of drawings
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=10, early_exaggeration=100)
X_tsne = tsne.fit_transform(X)

#remaps X-tsne 0-1 both directions
x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
X_tsne = (X_tsne - x_min) / (x_max - x_min)


overall_canvas_size = 1400
margin = 150
drawing_size = overall_canvas_size-2*margin
sketch_size = 70
point_size = 5


#generates base canvas
canvas =Image.new('RGB',(overall_canvas_size,overall_canvas_size), color = 'white')

#adds images
for i in range(0,len(list_directory[1])):
    sketch_filename = os.path.join(overall_results_directory, list_directory[1][i].replace('npy','jpg'))
    im_source=cv2.imread(sketch_filename)
    im_source_pil = Image.fromarray(im_source)
    im_source_pil = im_source_pil.resize((sketch_size,sketch_size), Image.ANTIALIAS)
    x = int(X_tsne[i][0]*drawing_size + margin-sketch_size/2)
    y = int(X_tsne[i][1]*drawing_size + margin-sketch_size/2)
    canvas.paste(im_source_pil,(x,y))


#savs file
tsne_filename_2 = os.path.join(overall_results_directory, 'tsne_images_' +  item + '.jpg')
canvas.save(tsne_filename_2)


