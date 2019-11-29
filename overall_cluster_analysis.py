
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
from project_data import website_colour
# ------------------------------------------------------------------------------------
# Imports locally defined functions
# ------------------------------------------------------------------------------------  
from drawing_app_functions import generate_image, draw_land_use_analysis, report_land_use
from imagenet_utils import preprocess_input
from graph_form_image import path_graph
from overall_analysis import basic_line_drawing, bundle_drawing
from tSNE import plot_tsne
from basic_drawing_functions import pts_to_polylines, draw_paths, draw_paths_base

#%%
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
 

#----------------------------------------------------------------------------------------
# calls image list and goes thogugh all images to generate images and feature files
#----------------------------------------------------------------------------------------
def feature_generation(exercise):
    item = item_list[exercise]
    item_number = item_number_list [exercise]
    
    #loads model
    list_directory = clean_line_list_dir()    
    print("Loading VGG19 pre-trained model...")
    base_model = VGG19(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
    
    #develops features form images
    imgs, X = [], []
    for sketch in list_directory[item_number]:
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
    
    feature_file_name = os.path.join(overall_results_directory, 'overall_features_' + item + '.npy')
    image_file_name = os.path.join(overall_results_directory, 'overall_images_' + item + '.npy')
    # appends the featueres and images into large file
    X = np.array(X)  # feature vectors
    np.save(feature_file_name, X)
    imgs = np.array(imgs)  # images
    np.save(image_file_name, imgs)


#----------------------------------------------------------------------------------------
#develps Tsne of drawings
#----------------------------------------------------------------------------------------
def tsne_plot(exercise):
    item = item_list[exercise]
    item_number = item_number_list [exercise]
    #loads files
    feature_file_name = os.path.join(overall_results_directory, 'overall_features_' + item + '.npy')
    X = np.load(feature_file_name )
    # loads parameters for tsne
    perplexity = perplexity_list[exercise]
    early_exaggeration = early_exaggeration_list[exercise] 
    #develops tsne
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity, early_exaggeration=early_exaggeration)
    X_tsne = tsne.fit_transform(X)
    #remaps X-tsne to 0-1 in x,y
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)
    
    #develops scatergraph with images image
    overall_canvas_size = 1400
    margin = 150
    drawing_size = overall_canvas_size-2*margin #  area where drawings will be circumscribed
    if len(list_directory[item_number]) < 40: # defining sketch size acccording to number
        sketch_size = 150
    else:
        if len(list_directory[item_number]) < 100:
            sketch_size = 80
        else: 
            sketch_size = 40  
            
    canvas =Image.new('RGB',(overall_canvas_size,overall_canvas_size), color = 'white') # generates base canvas  
    for i in range(0,len(list_directory[item_number])): # adds resized images into it
        sketch_filename = os.path.join(overall_results_directory, list_directory[item_number][i].replace('npy','jpg'))
        im_source=cv2.imread(sketch_filename)
        im_source=cv2.cvtColor(im_source, cv2.COLOR_BGR2RGB)
        im_source_pil = Image.fromarray(im_source)
        im_source_pil = im_source_pil.resize((sketch_size,sketch_size), Image.ANTIALIAS)
        x = int(X_tsne[i][0]*drawing_size + margin-sketch_size/2)
        y = int(X_tsne[i][1]*drawing_size + margin-sketch_size/2)
        canvas.paste(im_source_pil,(x,y))
    
    #saves file
    tsne_filename_2 = os.path.join(overall_results_directory, 'tsne_images_' +  item + '.jpg')
    canvas.save(tsne_filename_2)



#%%
list_directory = clean_line_list_dir()       


#%%


exercise = 1 # 0: lines, 1: massing
item_list=['lines','massing']
item_number_list = [1,4]
perplexity_list = [5,10]
early_exaggeration_list = [100,100]
tsne_plot(1)
#%%

exercise = 0
item = item_list[exercise]
item_number = item_number_list [exercise]
#loads files
feature_file_name = os.path.join(overall_results_directory, 'overall_features_' + item + '.npy')
image_file_name = os.path.join(overall_results_directory, 'overall_images_' + item + '.npy')
X = np.load(feature_file_name)
print (X.shape)
print(len(list_directory[item_number]))
