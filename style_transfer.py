import tensorflow as tf
import numpy as np
import cv2
import utils
import os
from matplotlib import pyplot as plt #used for debuggin purposes


from basic_drawing_functions import pts_to_polylines

from project_data  import node_coords_detailed, link_base_image, link_base_image_warning, model_directory, shape_y
from project_data  import shape_x, thickness_lines, color_canvas_rgb, content_target_resize_list, model_list


# ------------------------------------------------------------------------------------
# Estimates bounding square of site in order to crop image and style transfer only central area
# ------------------------------------------------------------------------------------
def bounding_box(points): 
    # Estimates bounding rectangle 
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)
    
    # fits smallest square centred int he same position of the rectangle
    rectangle_lengths = [top_right_x  - bot_left_x , top_right_y - bot_left_y]
    square_size = max(rectangle_lengths)
    rectangle_centre = [(top_right_x  + bot_left_x)/2 , (top_right_y + bot_left_y)/2]
    square_coordinates = ((rectangle_centre[0]-square_size/2),(rectangle_centre[1]-square_size/2),
                          (rectangle_centre[0]+square_size/2),(rectangle_centre[1]+square_size/2))
    return square_coordinates, square_size


# ------------------------------------------------------------------------------------
# Loads frozen graph for style transfer model
# ------------------------------------------------------------------------------------
def load_graph(frozen_graph_path):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
    return graph


# ------------------------------------------------------------------------------------
# Reads an image and develops a style transfer
# ------------------------------------------------------------------------------------
def stylize_image (frozen_graph_path, img, content_target_resize):
    # develop s astyle transfer of an image
    # Preprocess input image.
    img = utils.imresize(img, content_target_resize)
    img_4d = img[np.newaxis, :]

    # Load Graph
    graph=load_graph(frozen_graph_path)
    x = graph.get_tensor_by_name('img_t_net/input:0')
    y = graph.get_tensor_by_name('img_t_net/output:0')
    
    # Produces style transfer of image
    with tf.Session(graph=graph) as sess:
        img_out = sess.run(y, feed_dict={x: img_4d})
        print ('Saving image.')
        img_out = np.squeeze(img_out)
        img_out = cv2.resize(img_out, dsize=(700, 700), interpolation=cv2.INTER_CUBIC)
    return img_out


# ------------------------------------------------------------------------------------
# Generates a montage of lines and buildings over the style tarsnfer base in combinaiton with montage_image
# ------------------------------------------------------------------------------------
def call_montage(data, session_folder, file_name, folder_name):
    #checks model and content_target_resize and calls montage_image to carry out work
    #separated in order to allow tests with montage sepparately
    data=int(data[0][0])
    model_name = model_list[data]
    content_target_resize = content_target_resize_list[data]
    montage_image (session_folder, file_name, folder_name, model_name, content_target_resize)


# ------------------------------------------------------------------------------------
# Generates a montage of lines and buildings over the style tarsnfer base in combinaiton with call_montage
# ------------------------------------------------------------------------------------
def montage_image (session_folder, file_name, folder_name, model_name, content_target_resize):
    # kept separated from call montage in order to carry out tests with a number of models
    # develops a montage of stylized back over the actual site including overlaps, transparency, cropping and line drawing on top
    # Loads roads and buildings from folder
    filepath_np = os.path.join(session_folder, folder_name + '_lines.npy')
    filepath_np_massing = os.path.join(session_folder, folder_name + '_massing.npy')
    filepath_linetype_np = os.path.join(session_folder, folder_name + '_lines_type.npy')
    filepath_linetype_np_massing = os.path.join(session_folder, folder_name + '_massing_type.npy')
    
    points_massing = np.load(filepath_np_massing)
    line_type_massing=np.load(filepath_linetype_np_massing)
    points_lines = np.load(filepath_np)
    line_type_lines = np.load(filepath_linetype_np)
    
    #tests that arrays are not empty
    test = (len(points_massing[0])>0)*(len(points_lines[0])>0)
    if test == 1:
        #generates image to pass to neural
        line_type = np.concatenate((line_type_lines,line_type_massing),axis=0).tolist() # appends line types already drawn for paths
        points =  np.concatenate((points_lines,points_massing),axis=1).tolist() # appends lines already drawn for paths
        polylines  = pts_to_polylines(points, line_type)[0]
        line_type = pts_to_polylines (points, line_type) [1]
        
        #line drawing over blank  for both pats and buildings
        image = np.zeros((int(shape_x),int(shape_y),3), np.uint8)
        image.fill(255)
        for i in range(0, len(polylines)):
            thickness = thickness_lines[line_type[i]]
            color = (0,0,0) # all  to black for style transfer base
            for j in range(0, int(len(polylines[i])-1)):
                cv2.line(image,(int(polylines[i][j][0]),int(shape_y-polylines[i][j][1])),(int(polylines[i][j+1][0]),int(shape_y-polylines[i][j+1][1])),color,thickness)
        
        
        #image = draw_paths (pols, linetype, session_folder, file_name)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image=cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # makes grayscale to pass to neural for park design 
        
        #generates mask
        #white canvas
        img = np.zeros((700,700,3), np.uint8)
        img.fill(255)
        
        #draws polygon of site from nodes in edges which is later used as mask
        pts=np.array(node_coords_detailed, np.int32)
        pts=pts.reshape((-1,1,2))
        cv2.fillPoly(img,[pts],(0,0,0))
    
        #generates the mask
        mask=img ==0
        mask2=img!=0     
    
        #generates the base painting sending image to neural
        # identifies region of interest
        coords= bounding_box(node_coords_detailed)[0]
        original_size = bounding_box(node_coords_detailed)[1]
        
        # selects ROI from drawing, rescales to 700x700 and feeds to style trasnfer
        cropped_image = image[int(coords[0]):int(coords[2]),int(coords[1]):int(coords[3])]
        cropped_image = cv2.resize(cropped_image, dsize=(700, 700), interpolation=cv2.INTER_CUBIC)

        # sends to neural        
        frozen_graph_path = os.path.join(model_directory, model_name + str(content_target_resize) + '.pb')        
        cropped_image = stylize_image (frozen_graph_path, cropped_image, content_target_resize)
        
        # scales stylized image back and pastes over plan
        cropped_image = cv2.resize(cropped_image, dsize=(original_size, original_size), interpolation=cv2.INTER_CUBIC)
        image[int(coords[0]):int(coords[2]),int(coords[1]):int(coords[3])]=cropped_image
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # develops transparency
        # generates a white canvas to blend and simulate transparency
        img3 = np.zeros((700,700,3), np.uint8)
        img3.fill(255)
        transparency = 0.5
        image = cv2.addWeighted(img3,transparency,image,(1-transparency),0)    

        #draws lines over the stylized base
        for i in range(0, len(polylines)):
            thickness = thickness_lines[line_type[i]]
            color = color_canvas_rgb[line_type[i]]
            for j in range(0, int(len(polylines[i])-1)):
                cv2.line(image,(int(polylines[i][j][0]),int(shape_y-polylines[i][j][1])),(int(polylines[i][j+1][0]),int(shape_y-polylines[i][j+1][1])),color,thickness)
      
        #carries out the montage cropping the stylized image with the site and pasting it into the plan
        #plt.imshow(image, cmap="hot") plt can be used to show partial result for debuggin purposes
        img2=image*mask
        base=cv2.imread(link_base_image)
        base=base*mask2
        result = base + img2
        result=cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        #exports drawing for filing
        output_img_path = os.path.join(session_folder, file_name + '_stylized_montage.jpg')
        cv2.imwrite(output_img_path,result)    
    
    # If any of the arrays are empty trows error into the feedback image
    else:
        base=cv2.imread(link_base_image_warning)
        output_img_path = os.path.join(session_folder, file_name + '_stylized_montage.jpg')
        cv2.imwrite(output_img_path,base)


#%%
# test variables
#
#session_user =  '1573493802603'
#millis = 1573494063738
#file_name = session_user+'_'+ str(millis)
#root_data = 'D:\\GitHub_clones\\Drawscapes\\data\\'
#session_folder=os.path.join(root_data, session_user)
#folder_name = session_user
#
#
#model_name ='style_source_24_1_final'
#
#
#
#style_base_list = [33,35,4,15,24,26]
#scale_base_list = [1,5,10,15]
#content_target_resize_variables = [0.1, 0.3, 0.5, 0.8]
##
##
#for style_base in style_base_list:
#    for scale_base in scale_base_list:
#        for content_target_resize in content_target_resize_variables:
#            model_name = 'style_source_' + str(style_base) + '_' + str(scale_base) + '_final'     
#            file_name = model_name + str(content_target_resize)
#            print(model_name)
#            montage_image (session_folder, file_name, folder_name, model_name, content_target_resize)

#
