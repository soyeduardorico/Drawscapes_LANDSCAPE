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
from tinydb import TinyDB, Query

# ------------------------------------------------------------------------------------
# Imports project variables
# ------------------------------------------------------------------------------------
from project_data import link_base_image, link_base_image_large_annotated, link_base_image_warning
from project_data import ucl_east_image, databse_filepath, link_feedback_massing_base
from project_data import shape_y, shape_x, thickness_lines, root_participation_directory
from project_data import reference_directory_images, reference_directory, overall_results_directory
from project_data import link_outcome_failure, link_outcome_success, node_coords, threshold_distance
from project_data import node_coords_large, color_canvas_rgb, node_coords_detailed
from project_data import site_scale_factor, massing_height
from project_data import ucl_east_development_area, ucl_east_student_population, ucl_east_research_area
from project_data import ratio_accomodation_base, ratio_accomodation_plinth, ratio_accomodation_tower, m2accomodation_per_student
from project_data import ratio_research_base, ratio_research_plinth, ratio_research_tower
import project_data as pdt


# ------------------------------------------------------------------------------------
# Imports locally defined functions
# ------------------------------------------------------------------------------------
from imagenet_utils import preprocess_input
from graph_form_image import path_graph
from overall_analysis import basic_line_drawing, bundle_drawing
from database_management import data_to_database, line_data_from_database, export_data
from basic_drawing_functions import site_area, pts_to_polylines, draw_paths, draw_paths_base, draw_base_large

# -------------------------------------------------
# scale data pre calculation
# -------------------------------------------------

#generates data for scaling drawing from normal scale to wider scale (translation and scale)
#large scale drawing will be a smaller drawing since it shows a larger area of the city so scale_factor < 1
translate_move= [(node_coords_large[0][0]-node_coords[0][0]),(node_coords_large[0][1]-node_coords[0][1])]

max1=node_coords[0][0]
min1=node_coords[0][0]
for i in node_coords:
    if i [0]< min1:
        min1=i[0]
    if i[0]>max1:
        max1=i[0]

max2=node_coords_large[0][0]
min2=node_coords_large[0][0]
for i in node_coords_large:
    if i [0]< min2:
        min2=i[0]
    if i[0]>max2:
        max2=i[0]

scale_factor=(max2-min2)/(max1-min1)


# -------------------------------------------------
# Variables for large scale reports generation
# -------------------------------------------------
padding=60
thumb_x = shape_x
thumb_y = shape_y
report_grid_x=3
report_grid_y=3
canvas_x = report_grid_x*thumb_x + (report_grid_x-1)*padding
canvas_y = report_grid_y*thumb_y + (report_grid_y-1)*padding


# ------------------------------------------------------------------------------------
# File locations
# ------------------------------------------------------------------------------------
absFilePath = os.path.dirname(__file__)
root_data = os.path.join(absFilePath,  'data')



# ------------------------------------------------------------------------------------
# Generates main feedback for connectivity and style analysis
# ------------------------------------------------------------------------------------
def drawscapes_feedback_lines (data, file_name, session_folder, folder_name):
    data_1 = data.copy() # needs to make a copy otherwise with first pass of function it gets truncated
    exercise = pdt.exercises[0]

    # sends data to user and general databases
    export_data (data, folder_name, file_name, session_folder, exercise)

    # Generates images to be later served to front end
    generate_image (data_1, session_folder, file_name, folder_name)


# ------------------------------------------------------------------------------------
# Closes line drawing exercise, saves data and generates image base for massing exercise
# ------------------------------------------------------------------------------------
def drawscapes_draw_base (data, exercise, file_name, session_folder, folder_name):
    if len(data[4]) > 1 :
        data_1 = data.copy() # needs to make a copy otherwise with first pass of function it gets truncated
        # sends data to user and general databases
        export_data (data_1, folder_name, file_name, session_folder, exercise)

        # generates drawing base
        line_type = data[4]
        data.pop(4) # removes linetype array
        data.pop(0) # removes style array (one element) from list
        ptexport=np.array(data).astype(int)  # turn into integer since mobile devices will produce fractions and pythonanywhere saves as float
        points=ptexport.tolist()
        pts_to_polylines_list  = pts_to_polylines(points, line_type)
        polylines  = pts_to_polylines_list [0]
        linetype = pts_to_polylines_list [1] 
        img=cv2.imread(link_base_image)
        draw_paths_base (polylines, linetype, session_folder, file_name, img)
    else:
        base=cv2.imread(link_base_image_warning)
        b=os.path.join(session_folder,file_name +'_base'+'.jpg')
        cv2.imwrite(b,base)



# ------------------------------------------------------------------------------------
# Generates images for bases coming from the feedback session reading data from the database
# ------------------------------------------------------------------------------------
def drawscapes_draw_base_from_feedback (database, exercise, file_name, session_folder, user_id):
    img=cv2.imread(pdt.link_base_image)

    #begin with lines
    current_exercise = pdt.exercises[0]
    data_import = line_data_from_database(database, user_id, current_exercise)
    polylines = data_import[0]
    linetype= data_import[1]
    if len (polylines) > 1: # checks that there is actually data
        img=draw_paths_base (polylines, linetype, 'any', 'any', img, save='False')
    else:
        img= cv2.imread(pdt.draw_no_lines_drawn) # loads error file if no lines included)

    #follows with buildings if exercise = 1
    if exercise == 1:
        current_exercise = pdt.exercises[1]
        data_import = line_data_from_database(database, user_id, current_exercise)
        polylines = data_import[0]
        linetype= data_import[1]
        if len (polylines) > 1: # checks that there is actually data
            img=draw_paths_base (polylines, linetype, 'any', 'any', img, save='False')
        else:
            img= cv2.imread(pdt.draw_no_lines_drawn) # loads error file if no lines included)
        image_name = os.path.join(session_folder, file_name + '_landscape_base.jpg')
        cv2.imwrite(image_name,img)

    image_name = os.path.join(session_folder, file_name + '_base.jpg')
    cv2.imwrite(image_name,img)


# ------------------------------------------------------------------------------------
# Generates image base for land use exercise based on massing + paths
# brings buildings data as json from the front end but has to look for paths from the website sice thee were deleted when moving onto step 5
# ------------------------------------------------------------------------------------
def drawscapes_draw_base_for_land_use (data, file_name, session_folder, folder_name):
    file_name = file_name + '_landscape'
    if len(data[4]) > 1:
        # sends data to user and general databases
        exercise = pdt.exercises[1]
        data_1 = data.copy() # needs to make a copy otherwise with first pass of function it gets truncated
        export_data (data_1, folder_name, file_name, session_folder, exercise)

        #imports lines from database and develops a base
        data_import = line_data_from_database(databse_filepath, folder_name,'lines')
        polylines_0 = data_import[0]
        linetype_0= data_import[1] # needs reconverting to integer from float in database
        img=cv2.imread(link_base_image)
        img=draw_paths_base (polylines_0, linetype_0, 'any', 'any', img, save='False')

        #draws massing lines over base
        line_type = data[4]
        data.pop(4) # removes linetype array
        data.pop(0) # removes style array (one element) from list
        ptexport=np.array(data).astype(int)  # turn into integer since mobile devices will produce fractions and pythonanywhere saves as float
        points=ptexport.tolist()
        pts_to_polylines_list  = pts_to_polylines(points, line_type)
        polylines  = pts_to_polylines_list [0]
        linetype = pts_to_polylines_list [1]        
        img=draw_paths_base (polylines, linetype, session_folder, file_name, img, save='True')
    else:
        base=cv2.imread(link_base_image_warning)
        b=os.path.join(session_folder,file_name +'_base.jpg')
        cv2.imwrite(b,base)


# ------------------------------------------------------------------------------------
# Saves land uses data before moving to feedback stage. Saves image for further visual inspection
# ------------------------------------------------------------------------------------
def save_land_uses (data, session_folder, file_name, folder_name):
    if len(data[4]) > 1 :
        # sends data to user and general databases
        exercise = pdt.exercises[2]
        data_1 = data.copy() # needs to make a copy otherwise with first pass of function it gets truncated
        export_data (data_1, folder_name, file_name, session_folder, exercise)

        # generates image for record
        # imports lines from database and develops a base
        data_import = line_data_from_database(databse_filepath, folder_name,pdt.exercises[0])
        polylines_0 = data_import[0]
        linetype_0= data_import[1] # needs reconverting to integer from float in database
        img=cv2.imread(link_base_image)
        img=draw_paths_base (polylines_0, linetype_0, 'any', 'any', img, save='False')

        # imports massing from database and develops a base on top of existing drawing
        data_import = line_data_from_database(databse_filepath, folder_name,pdt.exercises[1])
        polylines_0 = data_import[0]
        linetype_0= data_import[1] # needs reconverting to integer from float in database
        img=draw_paths_base (polylines_0, linetype_0, 'any', 'any', img, save='False')

        # finalises drawing with land uses
        line_type = data[4]
        data.pop(4) # removes linetype array
        data.pop(0) # removes style array (one element) from list
        ptexport=np.array(data).astype(int)  # turn into integer since mobile devices will produce fractions and pythonanywhere saves as float
        points=ptexport.tolist()
        pts_to_polylines_list  = pts_to_polylines(points, line_type)
        polylines  = pts_to_polylines_list [0]
        linetype = pts_to_polylines_list [1]
        file_name = file_name + '_land_uses'
        img=draw_paths_base (polylines, linetype, session_folder, file_name, img, save='True')


# ------------------------------------------------------------------------------------
# Saves survey results
# ------------------------------------------------------------------------------------
def save_survey_results (data, session_folder, file_name, user_id):
    # global database
    database = pdt.databse_filepath
    db = TinyDB(database)
    survey_query=Query()
    #update / instert (upsert)
    db.upsert( {'id': user_id, 'survey_results' : data}, survey_query.id == user_id) # inserts or udates
    db.close()

    #user database
    database = os.path.join(session_folder, user_id + '_database.json')
    db = TinyDB(database)
    survey_query=Query()
    #update / instert (upsert)
    db.upsert( {'id': file_name, 'survey_results' : data}, survey_query.id == file_name) # inserts or udates
    db.close()
# ------------------------------------------------------------------------------------
# develops pixel count on drawing going through the list of land thickness used in massing_analysis
# ------------------------------------------------------------------------------------
def count_pixels (img):
    land_use = []
    built_area = 0
    for i in range (0, len(color_canvas_rgb)):
        sought = color_canvas_rgb[i]
        result = np.count_nonzero(np.all(img==sought,axis=2)) * site_scale_factor * site_scale_factor * massing_height[i]
        land_use.append(result)
        built_area = built_area + result
    return built_area, land_use

# ------------------------------------------------------------------------------------
# Develops massing calculations from drawn polylines
# ------------------------------------------------------------------------------------
def massing_analysis (polylines, linetype):
    # generates base drawing to count
    img = np.zeros((int(shape_x),int(shape_y),3), np.uint8)
    img.fill(255)
    img =draw_paths_base (polylines, linetype, 'any', 'any', img, save='False')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # develops masked drawings for proximity indicators
    img_base_canal_calculation= cv2.imread(pdt.base_canal_calculation)
    mask=img_base_canal_calculation <255
    img_canal_calculation = img*mask

    # develops masked drawings for proximity indicators
    img_base_noise_calculation= cv2.imread(pdt.base_noise_calculation)
    mask=img_base_noise_calculation <255
    img_noise_calculation = img*mask

    # develops pixel count on drawing going through the list of land thickness
    built_area= count_pixels(img)[0]
    land_use = count_pixels(img)[1]
    FAR = built_area/site_area()
    area_base = land_use[1]
    area_plinth = land_use[2]
    area_tower = land_use[3]

    area_tower_canal = count_pixels(img_canal_calculation)[1][3]
    area_tower_noise = count_pixels(img_noise_calculation)[1][3]

    # develops ratios
    area_accomodation = area_base*ratio_accomodation_base + area_plinth*ratio_accomodation_plinth + area_tower*ratio_accomodation_tower
    students = area_accomodation/m2accomodation_per_student
    area_research = area_base*ratio_research_base + area_plinth*ratio_research_plinth + area_tower*ratio_research_tower

    if area_tower>0:
        ratio_tower_canal = area_tower_canal /area_tower
        ratio_tower_noise = area_tower_noise /area_tower
    else:
        ratio_tower_canal = 0
        ratio_tower_noise = 0

    return land_use, built_area, FAR, area_accomodation, students, area_research, ratio_tower_canal, ratio_tower_noise, area_tower


# ------------------------------------------------------------------------------------
# Generates image with calculations which is served back to the feedback
# ------------------------------------------------------------------------------------
def drawscapes_feedback_massing (data, file_name, user_id):
    if len(data[4]) > 1 :
        # sends data to user and general databases
        exercise = pdt.exercises[1]
        data_1 = data.copy() # needs to make a copy otherwise with first pass of function it gets truncated
        folder_name = user_id
        session_folder = os.path.join(root_data, user_id)
        export_data (data_1, folder_name, file_name, session_folder, exercise)

        #brings data and generates drawing with land use analysis
        line_type = data[4]
        data.pop(4) # removes linetype array
        data.pop(0) # removes style array (one element) from list
        ptexport=np.array(data).astype(int)  # turn into integer since mobile devices will produce fractions and pythonanywhere saves as float    
        line_type_export = np.array(line_type).astype(int)

        # builds up point data and calls massing analysis
        points=ptexport.tolist()
        polylines = pts_to_polylines (points, line_type) [0]
        linetype = pts_to_polylines (points, line_type) [1]
        data_land_use = massing_analysis (polylines, linetype)

        # generate new canvas
        canvas =Image.open(link_feedback_massing_base)

        # instantiates class for text. Uses larger text than other since information is lower
        sz=20
        font_small = ImageFont.truetype("arial.ttf", size = sz)
        sz2=60
        font_large = ImageFont.truetype("arial.ttf", size = sz2)
        draw = ImageDraw.Draw(canvas)
        line_separation  = 30
        block_separation = 221

        # General massing feedback
        draw.text((217, 205),f"{ucl_east_development_area:,}" + ' m2',(157,195,230),font=font_large)
        draw_area = int(data_land_use[1])
        draw.text((217, 380),f"{draw_area:,}" + ' m2',(157,195,230),font=font_large)
        ratio_built = data_land_use[1] / ucl_east_development_area*100
        if ratio_built > 150:
             text1 = "You drew too much..... try reducing a bit to hit the target"
        else:
            if ratio_built < 75:
                text1 = "You did not draw enough..... try adding a bit to hit the target"
            else:
                text1 = "You drew something pretty close to the target!!!"
        draw.text((100, 550),text1,(255,255,0),font=font_small)

        # saves file
        b=os.path.join(session_folder,file_name +'_massing'+'.jpg')
        canvas.save(b)
    else:
        base=cv2.imread(link_base_image_warning)
        b=os.path.join(session_folder,file_name +'_massing'+'.jpg')
        cv2.imwrite(b,base)


# ----------------------------------------------------------------------------------
# Calls for development of skeletons and basic lines
# ----------------------------------------------------------------------------------
def draw_skeleton_graphs (img, session_folder, file_name, folder_name):
    # preprocessing of drawing = geenration of basic skeleton
    grey=skimage.img_as_ubyte(skimage.color.rgb2grey(img))
    binary=grey<250
    skel= morphology.skeletonize(binary)
    skel=skel*1

    #calls for development of dual graph and simplified lines bsaed on  skeleton
    sketch_grap=path_graph(skel,node_coords,threshold_distance,file_name,session_folder,folder_name, link_base_image,shape_x,shape_y)
    # sketch_grap.draw_basic_connections() # not requiredforthe time being
    sketch_grap.draw_graph()


# ----------------------------------------------------------------------------------
# Generates images to be served as feedback from style and connectivty analysis
# ----------------------------------------------------------------------------------
def generate_image (data, session_folder, file_name, folder_name):
    declared_style=int(data[0][0]) # reads first item on the list and turns into integer
    line_type = data[4]
    data.pop(4) # removes linetype array
    data.pop(0) # removes style array (one element) from list
    ptexport=np.array(data).astype(int)  # turn into integer since mobile devices will produce fractions and pythonanywhere saves as float
    line_type_export = np.array(line_type).astype(int)
    points=ptexport.tolist()
    pts_to_polylines_list  = pts_to_polylines(points, line_type)
    polylines  = pts_to_polylines_list [0]
    linetype = pts_to_polylines_list [1]

    img=draw_paths (polylines, linetype, session_folder,file_name) # draws paths in the small scale drawing ovwer white canvas for further processing
    img2=cv2.imread(link_base_image)
    draw_paths_base (polylines, linetype, session_folder,file_name,img2) # Draws paths in the small scale base

    # will develop connectivity even for style to ensure final drawing considered at the end
    draw_skeleton_graphs(img, session_folder,file_name, folder_name)





#%%
    
# ------------------------------------------------------------------------------------
# Tester
# ------------------------------------------------------------------------------------  



# session_user =  '1576064564452'
# millis = 1576064636649

# root_data = root_participation_directory
# session_folder=os.path.join(root_data, session_user)
# folder_name = session_user
# file_name= session_user + '_' + str(millis) + '_test'


# filepath_np_massing = os.path.join(session_folder, session_user + '_massing.npy')
# filepath_np_massing_type = os.path.join(session_folder, session_user + '_massing_type.npy')

# ptexport=np.load(filepath_np_massing).astype(int)
# points=ptexport.tolist()

# line_type=np.load(filepath_np_massing_type).astype(int)

# polylines  = pts_to_polylines(points, line_type)[0]
# linetype = pts_to_polylines (points, line_type) [1]


# #generates data

# style_save = [4]
# data=[]
# data.append(style_save)
# data.append(points[0])
# data.append(points[1])
# data.append(points[2])
# data.append(line_type.tolist())

# print(data)

# user_id = session_user


# drawscapes_feedback_massing (data, file_name, user_id)


