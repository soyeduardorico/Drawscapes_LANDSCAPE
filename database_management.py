import os
import numpy as np
import cv2
import re
import time
from sklearn import manifold
from tinydb import TinyDB, Query
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg19 import VGG19
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt #used for debuggin purposes

# ------------------------------------------------------------------------------------
# Imports project variables
# ------------------------------------------------------------------------------------  
from project_data import link_base_image, link_base_image_large_annotated, link_base_image_warning, ucl_east_image
from project_data import shape_y, shape_x, thickness_lines, root_participation_directory
from project_data import reference_directory_images, reference_directory, overall_results_directory
from project_data import link_outcome_failure, link_outcome_success, node_coords, threshold_distance
from project_data import node_coords_large, color_canvas_rgb, node_coords_detailed
from project_data import site_scale_factor, massing_height, databse_filepath
from project_data import ucl_east_development_area, ucl_east_student_population, ucl_east_research_area

# ------------------------------------------------------------------------------------
# Imports locally defined functions
# ------------------------------------------------------------------------------------  
from basic_drawing_functions import draw_paths_base, pts_to_polylines
from imagenet_utils import preprocess_input


#%%
# Locates database to use and VGG19 loads model



#%%
#----------------------------------------------------------------------------------------
# Generate lists of np data of sketches in directory
#----------------------------------------------------------------------------------------
def list_files_dir():
    list_files = [] # file names
    sketches = os.listdir(overall_results_directory)
    #generates list of files
    for i in sketches:          
        if not i == 'Thumbs.db':
            if re.findall('_lines.npy',i) :
                list_files.append(i.replace('_lines.npy', ''))    
    return list_files

#----------------------------------------------------------------------------------------
# Generates VVG19 image extratced features from polylines
#----------------------------------------------------------------------------------------
def feature_extract (polylines, linetype):
    base_model = VGG19(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output) 
    print ('model loaded')
    img = np.zeros((int(shape_x),int(shape_y),3), np.uint8)
    img.fill(255)
    img = draw_paths_base (polylines, linetype, overall_results_directory, 'anyname', img, save='False')
    img = Image.fromarray(img)
    img = img.resize((224,224), Image.ANTIALIAS)
    img = image.img_to_array(img)  # convert to array
    img = np.expand_dims(img, axis=0)
    print('image geenrated')
    img = preprocess_input(img)
    print('image preprocessed')
    features = model.predict(img).flatten()  # features
    print('features extracted')   
    return features.tolist()


# ------------------------------------------------------------------------------------
# Sends sketch to feature_extract and uses the features to classify it 
# ------------------------------------------------------------------------------------
def style_classify (sketch_to_analyze):
    model_name=os.path.join(reference_directory, 'finalized_model.sav') 
    loaded_model = joblib.load(model_name)
    test_feature = feature_extract(sketch_to_analyze).reshape(1, -1)
    predicted_style=loaded_model.predict(test_feature)[0]
    return predicted_style

#----------------------------------------------------------------------------------------
# Reads image and type of exervice (massing or line), generates all fields and updates database (upsert method)
#----------------------------------------------------------------------------------------
def test_and_file (db, sketch, exercise, extract_features = 'True'):
    file_name = sketch
    filepath_np = os.path.join(overall_results_directory, file_name + '_' + exercise + '.npy')
    filepath_np_type = os.path.join(overall_results_directory, file_name + '_' + exercise + '_type.npy')
    if os.path.exists(filepath_np):
        ptexport=np.load(filepath_np).astype(int)
        points=ptexport.tolist()
        line_type=np.load(filepath_np_type).astype(int)
        pts_to_polylines_list =  pts_to_polylines(points, line_type)
        polylines  = pts_to_polylines_list [0]
        linetype = pts_to_polylines_list [1]
        if len(polylines[0])>1:
            field_polylines = exercise + '_polylines'
            field_linetype = exercise + '_linetype'
            field_features = exercise + '_features'
            print('processing data for ' + file_name)
            
            # in case we do not want to extract features we can bypass this
            if extract_features == 'True':
                feature_values=feature_extract (polylines, linetype)
            else:
                feature_values = []
                
            #brings data into database
            #turns data from 'int' to 'float' otherwise json will not like it
            linetype = [float(i) for i in linetype]            
            polylines[0]= np.array(polylines[0]).astype(float).tolist()
            sketch_query=Query()
            #update / instert (upsert)
            db.upsert( {'id': file_name, field_polylines : polylines, field_linetype : linetype, field_features : feature_values}, sketch_query.id == file_name) # inserts or udates

def data_to_database (databse_filepath, polylines, linetype, id_name, exercise, extract_features = 'True'):
    field_polylines = exercise + '_polylines'
    field_linetype = exercise + '_linetype'
    field_features = exercise + '_features'
    millis2 = int(round(time.time() * 1000))
    
    # in case we do not want to extract features we can bypass this
    if extract_features == 'True':
        feature_values=feature_extract (polylines, linetype)

    else:
        feature_values = []        
    linetype = [float(i) for i in linetype]            
    polylines[0]= np.array(polylines[0]).astype(float).tolist()
    db = TinyDB(databse_filepath)
    sketch_query=Query()
    #update / instert (upsert)
    db.upsert( {'id': id_name, field_polylines : polylines, field_linetype : linetype, field_features : feature_values}, sketch_query.id == id_name) # inserts or udates
    db.close()


#----------------------------------------------------------------------------------------
# Reads line data from database 
#----------------------------------------------------------------------------------------
def line_data_from_database (databse_filepath, user_id, exercise):
    db = TinyDB(databse_filepath)
    sketch_item=Query()
    field_polylines = exercise + '_polylines'
    field_linetype = exercise + '_linetype'
    data_polylines_dict = db.search(sketch_item.id==user_id)[0] # it brings a list of the dictinaries with that name, which in this case will be 1  long only
    if field_polylines in data_polylines_dict: # checks whether the field exists or the user skipped use
        data_polylines_float = data_polylines_dict.get(field_polylines)
        data_polylines = []
        for i in data_polylines_float:
            data_polylines.append(np.array(i).astype(int).tolist()) # needs reconverting to integer from float in database
        data_linetype = db.search(sketch_item.id==user_id)[0].get(field_linetype)
        data_linetype =  np.array(data_linetype).astype(int).tolist() # needs reconverting to integer from float in database
    else:
        data_polylines = [[[]]]
        data_linetype  =[]
    return data_polylines, data_linetype


#----------------------------------------------------------------------------------------
# Develops TSNE reading features and geometries from database for an exercise (lines =0 or massing =1)
#----------------------------------------------------------------------------------------
def tsne_embedding (db, exercise, id_name = 'anyname'):
    exercise_list=['lines','massing']
    perplexity_list = [5,10]
    early_exaggeration_list = [1,100]
    
    perplexity = perplexity_list[exercise]
    early_exaggeration = early_exaggeration_list[exercise] 
    
    extract_polylines = exercise_list[exercise] + '_polylines'
    extract_linetype  = exercise_list[exercise] + '_linetype'
    extract_features = exercise_list[exercise] + '_features'
    
    sketch_item=Query()
    if exercise == 0:
        db2 = db.search(sketch_item.lines_features != [])
    else:
        db2 = db.search(sketch_item.massing_features != [])
    
    number_items = len(db2)
    
    #reads VGG19 abstract featrues from db
    X=[]
    for item in db2:
        X.append(np.array(item.get(extract_features)))
    X = np.array(X) 

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
    
    if number_items < 40: # defining sketch size acccording to number
        sketch_size = 150
    else:
        if number_items < 100:
            sketch_size = 60
        else: 
            sketch_size = 40  
    
    circle_radius = int(sketch_size/2)
    draw_circle = 'False'    
    
    canvas =Image.new('RGB',(overall_canvas_size,overall_canvas_size), color = 'white') # generates base canvas  
    i=0
    for item in db2:
        polylines = item.get(extract_polylines)
        polylines[0]= np.array(polylines[0]).astype(int).tolist()
        linetype = np.array(item.get(extract_linetype)).astype(int)
        img = np.zeros((int(shape_x),int(shape_y),3), np.uint8)
        img.fill(255)
        file_name = 'any' #not required since no saving is performed
        img = draw_paths_base (polylines, linetype, overall_results_directory, file_name, img, save='False')
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_source_pil = Image.fromarray(img)
        im_source_pil = im_source_pil.resize((sketch_size,sketch_size), Image.ANTIALIAS)
        x = int(X_tsne[i][0]*drawing_size + margin-sketch_size/2)
        y = int(X_tsne[i][1]*drawing_size + margin-sketch_size/2)
        name = item.get('id')
        canvas.paste(im_source_pil,(x,y))
        if name == id_name: # check which is the item requested to draw circle after all images are pasted
            draw_circle = 'True'
            xc=x
            yc=y
        i=i+1
    if draw_circle == 'True': #draws circle aftera all images are pasted to avoid covering
        draw = ImageDraw.Draw(canvas)
        draw.ellipse((xc-circle_radius+sketch_size/2, yc-circle_radius+sketch_size/2, xc+circle_radius+sketch_size/2, yc+circle_radius+sketch_size/2), outline ='red')        
    
    tsne_filename_2 = os.path.join(overall_results_directory, 'tsne_images_' +  exercise_list[exercise] + '.jpg')
    canvas.save(tsne_filename_2)
   
    



#%%
    
#exercise = 1 # 0: lines, 1: massing    
#    

#db.purge()   
#for sketch in list_files_dir():
#    file_name = sketch
#    test_and_file (db, sketch, 'lines')
#    test_and_file (db, sketch, 'massing')


#%%


#%%
#file_name = '1574548541279'
#id_name = '1574548541279_test'
#
#exercise_number = 0
#exercise_list=['lines','massing']
#exercise = exercise_list[exercise_number]
#
#filepath_np = os.path.join(overall_results_directory, file_name + '_' + exercise + '.npy')
#filepath_np_type = os.path.join(overall_results_directory, file_name + '_' + exercise + '_type.npy')
#ptexport=np.load(filepath_np).astype(int)
#points=ptexport.tolist()
#line_type=np.load(filepath_np_type).astype(int)
#pts_to_polylines_list =  pts_to_polylines(points, line_type)
#polylines  = pts_to_polylines_list [0]
#linetype = pts_to_polylines_list [1]
#
#data_to_database (polylines, linetype, id_name, exercise, extract_features = 'True')
#
#print(len(db))
#tsne_embedding (db, exercise_number, id_name)

#%%
#sketch_item=Query()
#id_name = '1575016808957'
#for item in db:
#    features = item.get('lines_features')
#    print (len(features))
#
#
#db2 = db.search(sketch_item.lines_features != [])    
#for item in db2:
#    features = item.get('lines_features')
#    print (len(features))  
#    
    
    
    
#%%
# db = TinyDB(databse_filepath)
# id_name = '1575681037493'
# tsne_embedding (db, 0, id_name)
