import tensorflow as tf
import numpy as np
import cv2
import utils
import os
from matplotlib import pyplot as plt #used for debuggin purposes

from drawing_app_functions import pts_to_polylines

from project_data  import node_coords_detailed, link_base_image, model_directory, shape_y


def bounding_box(points):
    # Estimages bounding rectangle and 
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


def stylize_image (frozen_graph_path, img, content_target_resize):
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

def montage_image (data,session_folder, file_name, basic_file_name, model_name, content_target_resize):
    # Generates file names
    frozen_graph_path = os.path.join(model_directory, model_name + str(content_target_resize) + '.pb')
    input_img_path = os.path.join(session_folder, file_name + '.jpg')
    
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
    image = cv2.imread(input_img_path)
    
    # selects ROI from drawing, rescales to 700x700 and feeds to style trasnfer
    cropped_image = image[int(coords[0]):int(coords[2]),int(coords[1]):int(coords[3])]
    cropped_image = cv2.resize(cropped_image, dsize=(700, 700), interpolation=cv2.INTER_CUBIC)
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
        
    #Draws road over tranfered style
    #reads data on polylines
    
    # saves linetype as np
    file_path = os.path.join(session_folder, file_name + '_linetype.npy')         
    line_type=np.load(file_path).tolist()  

    points=data
    polylines  = pts_to_polylines(points, line_type)[0]
    
    # Going thoguh polylines twice with differtn colours
    thickness_lines_out = 7
    thickness_lines_in = 3
    for i in range(0, len(polylines)):
            for j in range(0, int(len(polylines[i])-1)):
                cv2.line(image ,(int(polylines[i][j][0]),int(shape_y-polylines[i][j][1])),(int(polylines[i][j+1][0]),int(shape_y-polylines[i][j+1][1])),(0,0,0),thickness_lines_out)
        
    for i in range(0, len(polylines)):
            for j in range(0, int(len(polylines[i])-1)):
                cv2.line(image ,(int(polylines[i][j][0]),int(shape_y-polylines[i][j][1])),(int(polylines[i][j+1][0]),int(shape_y-polylines[i][j+1][1])),(120,120,120),thickness_lines_in)
    
    #carries out the montage cropping the stylized image with the site and pasting it into the plan
    #plt.imshow(image, cmap="hot") plt can be used to show partial result for debuggin purposes
    img2=image*mask
    base=cv2.imread(link_base_image)
    base=base*mask2
    result = base + img2
    
    #exports drawing for filing
    output_img_path = os.path.join(session_folder, file_name + '_stylized_montage.jpg')
    cv2.imwrite(output_img_path,result)
    
    #exports drawing for drawscapes_massing.html canvas base
    output_img_path = os.path.join(session_folder, basic_file_name + '_stylized_base.jpg')
    cv2.imwrite(output_img_path,result)


#%%
#session_user =  '1573493802603'
#millis = 1573494040617
#file_name= str(millis)
#
#root_data = 'D:\\GitHub_clones\\Drawscapes\\data\\'
#session_folder=os.path.join(root_data, session_user)
#             
#content_target_resize_list = [0.1, 0.3, 0.5, 0.8]
#style_source_list = [33,36]
#style_target_resize_series_names = [1,5,10,15]
#
#model_name_list =[]
#for i in style_source_list :
#    for j in style_target_resize_series_names :
#        model_name_list.append('style_source_' + str(i) + '_' + str(j) + '_final')
#
#filepath_np = os.path.join(session_folder, file_name + '.npy')
#data=np.load(filepath_np).tolist()
#
#for model_name in model_name_list:
#    for content_target_resize in content_target_resize_list:
#        basic_file_name = model_name + '_' + str(content_target_resize)
#        montage_image (data, session_folder, file_name, basic_file_name, model_name, content_target_resize)


