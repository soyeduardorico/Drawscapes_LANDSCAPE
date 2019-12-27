#import os
import numpy as np
import cv2
import os
import tinydb
import skimage
from skimage import morphology

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from tinydb import TinyDB, Query
from matplotlib import pyplot as plt #used for debuggin purposes

# ------------------------------------------------------------------------------------
# Imports locally defined functions
# ------------------------------------------------------------------------------------
from database_management import line_data_from_database
from basic_drawing_functions import site_area, pts_to_polylines, draw_paths, draw_paths_base, draw_base_large
from drawing_app_functions import massing_analysis
from graph_form_image import path_graph
import project_data as pdt

# ------------------------------------------------------------------------------------
# File location
# ------------------------------------------------------------------------------------
absFilePath = os.path.dirname(__file__)
root_data = os.path.join(absFilePath,  'data')

def generate_image_feeback (img, text, text_colour, base_file_name, file_name, user_id, title ):
    dim = (530,530)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # resizes images
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sz=80
    font_large = ImageFont.truetype("arial.ttf", size = sz)
    img_pil = Image.fromarray(img)
    base_image_pil = Image.open(base_file_name) # brings larger canvas
    draw = ImageDraw.Draw(base_image_pil)
    draw.text((958, 411),text,text_colour,font=font_large)
    base_image_pil.paste(img_pil,(85,85))
    save_folder = os.path.join(root_data, user_id) # saves file
    feedback_filename = os.path.join(save_folder,file_name + title )
    base_image_pil.save(feedback_filename)

def obtain_connections (img):
    grey=skimage.img_as_ubyte(skimage.color.rgb2grey(img))
    binary=grey<250
    skel= morphology.skeletonize(binary)
    skel=skel*1
    skel_draw=1-skel

    #calls for development of dual graph and simplified lines bsaed on  skeleton
    file_name = 'any'
    session_folder = 'any'
    folder_name = 'any'
    sketch_grap=path_graph(skel,pdt.node_coords,pdt.threshold_distance, file_name, session_folder, folder_name, pdt.link_base_image, pdt.shape_x,pdt.shape_y)
    number_connections_under_bridge = sketch_grap.connections_under_bridge()
    return number_connections_under_bridge

def generate_feedback_images (databse_filepath, user_id, file_name):
    # load line data from different exercises
    exercise = pdt.exercises[0]  #import massing data for this feedback operation
    data_import = line_data_from_database(databse_filepath, user_id,exercise)
    polylines_lines = data_import[0]
    linetype_lines = data_import[1]

    exercise = pdt.exercises[1]  #import massing data for this feedback operation
    data_import = line_data_from_database(databse_filepath, user_id,exercise)
    polylines_massing = data_import[0]
    linetype_massing = data_import[1]

    exercise = pdt.exercises[2]  #import massing data for this feedback operation
    data_import = line_data_from_database(databse_filepath, user_id,exercise)
    polylines_land_uses = data_import[0]
    linetype_land_uses = data_import[1]

    massing_feedback_analysis = massing_analysis(polylines_massing, linetype_massing)

    # feedback on canal proximity
    if len (polylines_massing) > 1: # checks that there is actually data
        img=cv2.imread(pdt.feedback_canal_base)
        img=draw_paths_base (polylines_massing, linetype_massing, 'any', 'any', img, save='False')
        text = massing_feedback_analysis[6]
        text = str(int(text*100))+'%'
        text_colour= (1,168,80)
        generate_image_feeback (img, text, text_colour, pdt.feedback_canal, file_name,user_id, '_feedback_canal.jpg' )
    else:
        img= cv2.imread(pdt.draw_no_lines_drawn) # loads error file if no lines included
        generate_image_feeback (img, text, text_colour, pdt.feedback_canal, file_name, user_id, '_feedback_canal.jpg' )

    # feedback on noise impact proximity
    if len (polylines_massing) > 1:  # checks that there is actually data
        img=cv2.imread(pdt.feedback_noise_base)
        img=draw_paths_base (polylines_massing, linetype_massing, 'any', 'any', img, save='False')
        text = massing_feedback_analysis[7]
        text = str(int(text*100))+'%'
        text_colour= (230,0,170)
        generate_image_feeback (img, text, text_colour, pdt.feedback_noise, file_name,user_id, '_feedback_noise.jpg' )
    else:
        img= cv2.imread(pdt.draw_no_lines_drawn) # loads error file if no lines included
        generate_image_feeback (img, text, text_colour, pdt.feedback_noise, file_name,user_id,'_feedback_noise.jpg' )

    # feedback on barrier impact proximity
    if len (polylines_lines) > 1: # checks that there is actually data
        img=cv2.imread(pdt.feedback_barrier_base)
        img=draw_paths_base (polylines_lines, linetype_lines, 'any', 'any', img, save='False')
        text = obtain_connections(img)
        text = '  ' +  str(text)
        text_colour= (230,0,170)
        generate_image_feeback (img, text, text_colour, pdt.feedback_barrier, file_name,user_id, '_feedback_barrier.jpg' )
    else:
        img= cv2.imread(pdt.draw_no_lines_drawn) # loads error file if no lines included
        generate_image_feeback (img, text, text_colour, pdt.feedback_barrier, file_name,user_id, '_feedback_barrier.jpg' )


#%%
# TEST

#user_id = '1576685956770'
#millis = 1576686034588
#file_name= user_id + '_'+ str(millis)
#
#generate_feedback_images(pdt.databse_filepath, user_id, file_name)

#%%

#session_user =  '1576146133533'
#millis = '1576024145483_2'
#file_name = session_user  + '_' + millis
#root_data = root_participation_directory
#session_folder=os.path.join(root_data, session_user)
#folder_name = session_user
#
#
#image_file_path = os.path.join(session_folder, '1576146133533_1576146147920.jpg')
#img=cv2.imread(image_file_path)
#
#plt.imshow(img)
#print(img)
#grey=skimage.img_as_ubyte(skimage.color.rgb2grey(img))
#binary=grey<220
#skel= morphology.skeletonize(binary)
#skel_plot=skel*255
#
#
#plt.imshow(skel_plot)
#skel_plot=skimage.img_as_ubyte(skimage.color.grey2rgb(skel_plot))
#print(skel_plot)
#image_file_save_temporal = os.path.join(session_folder, 'temporal.jpg')
##skel = cv2.cvtColor(skel, cv2.COLOR_GREY2RGB)
#cv2.imwrite(image_file_save_temporal,skel_plot)

#%%

#sketch_grap=path_graph(skel,node_coords,threshold_distance,file_name,session_folder,folder_name, link_base_image,shape_x,shape_y)
#
#points1 = sketch_grap.key_points()
#pt = points1[0][0]
#print(pt)
#nodes = sketch_grap.nodes
#print(nodes)
#
#
#def snap(a):
#    closest = sketch_grap.closest_point(a,nodes)
#    distance = sketch_grap.dist_nodes(a, closest)
#    if distance < sketch_grap.threshold:
#        snap_point = closest
#    else:
#        snap_point = a
#    return snap_point
#
#
#def key_points():
#    end_point_list=[]
#    junction_list=[]        
#    for i in sketch_grap.pixel_graph().degree:
#        if i[1]==1:
#            pt =  snap(i[0])
#            end_point_list.append(pt)
#        if i[1]==3:
#            junction_list.append(i[0])        
#    return(end_point_list,junction_list)
#
#
#
#
#
#print(sketch_grap.key_points())
#%%



#sketch_grap.draw_graph()
#%%
    
# ------------------------------------------------------------------------------------
# Load data via file read
# ------------------------------------------------------------------------------------  



#session_user =  '1576024131225'
#millis = 1576024179366
#
#root_data = root_participation_directory
#session_folder=os.path.join(root_data, session_user)
#folder_name = session_user
#file_name= session_user + '_' + str(millis) + '_test'
#
#
#filepath_np_massing = os.path.join(session_folder, session_user + '_massing.npy')
#filepath_np_massing_type = os.path.join(session_folder, session_user + '_massing_type.npy')
#
#ptexport=np.load(filepath_np_massing).astype(int)
#points=ptexport.tolist()
#
#line_type=np.load(filepath_np_massing_type).astype(int)
#
#polylines  = pts_to_polylines(points, line_type)[0]
#linetype = pts_to_polylines (points, line_type) [1]
#
#
##generates data
#
#style_save = [4]
#data=[]
#data.append(style_save)
#data.append(points[0])
#data.append(points[1])
#data.append(points[2])
#data.append(line_type.tolist())
#
#print(data)
#
#user_id = session_user
#
#
#drawscapes_feedback_massing (data, file_name, user_id)
#
#
#
#padding=60



#%%


