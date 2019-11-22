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


# ------------------------------------------------------------------------------------
# Imports locally defined functions
# ------------------------------------------------------------------------------------  
from imagenet_utils import preprocess_input
from graph_form_image import path_graph
from overall_analysis import basic_line_drawing, bundle_drawing


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
# Draws polygon of site from nodes in edges for area estimates
# ------------------------------------------------------------------------------------
def site_area():
    # white canvas
    img= np.zeros((700,700,3), np.uint8)
    img.fill(255)
    
    # draws an outline with detailed nodes
    pts=np.array(node_coords_detailed, np.int32)    
    pts=pts.reshape((-1,1,2))    
    # fills site with black
    cv2.fillPoly(img,[pts],(0,0,0))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # calculates pixels
    sought = [0,0,0]
    result = np.count_nonzero(np.all(img==sought,axis=2))
    # rescles to m2
    area = result * site_scale_factor * site_scale_factor
    return area


# ------------------------------------------------------------------------------------
# Saves land uses data
# ------------------------------------------------------------------------------------
def save_land_uses (data, session_folder, file_name, folder_name):
    declared_style=int(data[0][0]) # reads first item on the list and turns into integer
    line_type = data[4]
    data.pop(4) # removes linetype array  
    data.pop(0) # removes style array (one element) from list
    ptexport=np.array(data).astype(int)  # turn into integer since mobile devices will produce fractions and pythonanywhere saves as float
    line_type_export = np.array(line_type).astype(int)
    
    # saves data as csv
    file_path = os.path.join(session_folder, file_name + '.csv')         
    np.savetxt(file_path, ptexport, delimiter=",")
    
    # saves data as numpy
    file_path = os.path.join(session_folder, file_name + '.npy')
    np.save(file_path, ptexport)
    
    # saves line and stlyle data as numpy with base name as most up to date option both in the folder and the overall results dir
    file_path = os.path.join(session_folder, folder_name + '_land_uses.npy')
    np.save(file_path, ptexport)
    file_path = os.path.join(overall_results_directory, folder_name + '_land_uses.npy')
    np.save(file_path, ptexport)        
    file_path = os.path.join(session_folder, folder_name + '_land_uses_type.npy')
    np.save(file_path, line_type_export)       
    file_path = os.path.join(overall_results_directory, folder_name + '_land_uses_type.npy')
    np.save(file_path, line_type_export)  


# ------------------------------------------------------------------------------------
# Generates main feedback for connectivity and style analysis
# ------------------------------------------------------------------------------------
def drawscapes_feedback_function (data, file_name, session_folder, folder_name, task):
    declared_style=int(data[0][0]) # reads first item on the list and turns into integer
    line_type = data[4]
    data.pop(4) # removes linetype array  
    data.pop(0) # removes style array (one element) from list
    ptexport=np.array(data).astype(int)  # turn into integer since mobile devices will produce fractions and pythonanywhere saves as float
    line_type_export = np.array(line_type).astype(int)
    
    # saves data as csv
    file_path = os.path.join(session_folder, file_name + '.csv')         
    np.savetxt(file_path, ptexport, delimiter=",")
    
    # saves data as numpy
    file_path = os.path.join(session_folder, file_name + '.npy')
    np.save(file_path, ptexport)
    
    # saves line and stlyle data as numpy with base name as most up to date option both in the folder and the overall results dir
    file_path = os.path.join(session_folder, folder_name + '_lines.npy')
    np.save(file_path, ptexport)
    file_path = os.path.join(overall_results_directory, folder_name + '_lines.npy')
    np.save(file_path, ptexport)        
    file_path = os.path.join(session_folder, folder_name + '_lines_type.npy')
    np.save(file_path, line_type_export)       
    file_path = os.path.join(overall_results_directory, folder_name + '_lines_type.npy')
    np.save(file_path, line_type_export)    
    # generates jpg image
    # 1 for style, 2 for connectivity. This will tell feedack function to not carry all tasks
    generate_image(ptexport, line_type, session_folder, file_name, folder_name, declared_style, task)


# ------------------------------------------------------------------------------------
# Generates image base for massing exercise  
# ------------------------------------------------------------------------------------
def drawscapes_draw_base (data, file_name, session_folder, folder_name):
    if len(data[4]) > 1 :
        line_type = data[4]
        data.pop(4) # removes linetype array  
        data.pop(0) # removes style array (one element) from list
        ptexport=np.array(data).astype(int)  # turn into integer since mobile devices will produce fractions and pythonanywhere saves as float
        points=ptexport.tolist()
        pols  = pts_to_polylines(points, line_type)[0]
        
        # saves line and stlyle data as numpy with base name as most up to date option both in the folder and the overall results dir
        line_type_export = np.array(line_type).astype(int)
        file_path = os.path.join(session_folder, folder_name + '_lines.npy')
        np.save(file_path, ptexport)    
        file_path = os.path.join(overall_results_directory, folder_name + '_lines.npy')
        np.save(file_path, ptexport)  
        file_path = os.path.join(session_folder, folder_name + '_lines_type.npy')
        np.save(file_path, line_type_export)           
        file_path = os.path.join(overall_results_directory, folder_name + '_lines_type.npy')
        np.save(file_path, line_type_export)         

        # generates drawing base
        draw_paths_base (pols, line_type, session_folder, file_name)
    else:
        base=cv2.imread(link_base_image_warning)
        b=os.path.join(session_folder,file_name +'_base'+'.jpg')
        cv2.imwrite(b,base)        


# ------------------------------------------------------------------------------------
# Generates image base for landscape exercise based on massing + paths
# brings buildings data as json from the front end but has to look for paths from the website sice thee were deleted when moving onto step 5
# ------------------------------------------------------------------------------------
def drawscapes_draw_base_2 (data, file_name, session_folder, folder_name):
    file_name = file_name + '_landscape'
    if len(data[4]) > 1 :
        line_type = np.array(data[4])
        filepath_np = os.path.join(session_folder, folder_name + '_lines_type.npy')
        line_type2=np.load(filepath_np).astype(int)
        line_type = np.concatenate((line_type,line_type2),axis=0).tolist() # appends line types already drawn for paths
        data.pop(4) # removes linetype array  
        data.pop(0) # removes style array (one element) from list
        ptexport=np.array(data).astype(int)  # turn into integer since mobile devices will produce fractions and pythonanywhere saves as float
        filepath_np = os.path.join(session_folder, folder_name + '_lines.npy')
        ptexport2 = np.load(filepath_np).astype(int)  
        ptexport =  np.concatenate((ptexport,ptexport2),axis=1) # appends lines already drawn for paths
        points=ptexport.tolist()
        pols  = pts_to_polylines(points, line_type)[0]
        linetype = pts_to_polylines (points, line_type) [1]
        draw_paths_base (pols, linetype, session_folder, file_name)
    else:
        base=cv2.imread(link_base_image_warning)
        b=os.path.join(session_folder,file_name +'_base.jpg')
        cv2.imwrite(b,base)  


# ------------------------------------------------------------------------------------
# Generates a list of polylines given a list of points
# ------------------------------------------------------------------------------------
def pts_to_polylines (points, line_type): 
    polylines=[]
    line=[]
    linetypes = [] # lsit of type of line for thickness and color definition
    #generates lines of polylines
    for i in range (0, int(len(points[0])-1)):
        if points[2][i+1] > 1:
            pt=[points[0][i],shape_y-points[1][i]]
            line.append(pt)
        else:
            pt=[points[0][i],shape_y-points[1][i]]
            line.append(pt)
            polylines.append(line)
            linetypes.append(line_type[i])
            line=[]
    polylines.append(line)
    linetypes.append(line_type[len(line_type)-1])
    return polylines, linetypes


def draw_paths (polylines, line_type, folder,file_name):
    #line drawing over blank canvas
    img = np.zeros((int(shape_x),int(shape_y),3), np.uint8)
    img.fill(255)
    for i in range(0, len(polylines)):
        thickness = thickness_lines[line_type[i]]
        color = color_canvas_rgb[line_type[i]]
        for j in range(0, int(len(polylines[i])-1)):
            cv2.line(img,(int(polylines[i][j][0]),int(shape_y-polylines[i][j][1])),(int(polylines[i][j+1][0]),int(shape_y-polylines[i][j+1][1])),color,thickness)
    b=os.path.join(folder, file_name + '.jpg')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(b,img2)
    return img2
    

def draw_paths_base (polylines, line_type, folder,file_name):    
    #line drawing over base
    img2=cv2.imread(link_base_image)
    for i in range(0, len(polylines)):
        thickness = thickness_lines[line_type[i]]
        color = color_canvas_rgb[line_type[i]]        
        for j in range(0, int(len(polylines[i])-1)):
            cv2.line(img2,(int(polylines[i][j][0]),int(shape_y-polylines[i][j][1])),(int(polylines[i][j+1][0]),int(shape_y-polylines[i][j+1][1])),color,thickness)
    b2=os.path.join(folder, file_name + '_base.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    cv2.imwrite(b2,img2)
    return img2


# ------------------------------------------------------------------------------------
# Draws simplified lines into the large scale plan of the site
# ------------------------------------------------------------------------------------
def draw_base_large (polylines, session_folder,file_name):
    b1=os.path.join(session_folder, file_name + '_large_overall.jpg')
    pos_x=0
    pos_y=0
    img = cv2.imread(link_base_image_large_annotated)
    #re-scales points in polylines: Displaces points from node[0][0] to origin, applies scale_factor and moves back to nodes_large [0][0]
    polylines2=[]
    line=[]
    for i in range (0, len(polylines)):
        for j in range (0, len(polylines[i])):
            a = (polylines[i][j][0] - node_coords[0][0]-pos_x)*scale_factor+node_coords_large[0][0]+pos_x
            b = (polylines[i][j][1] - (shape_y-node_coords[0][1])-pos_y)*scale_factor+(shape_y-node_coords_large[0][1])+pos_y            
            pt=[a,b]
            line.append(pt)
        polylines2.append(line)    
        line=[]    
    #interates over the polylines to draw them. Thickness is fixed to = 5
    for i in range(0, len(polylines2)):
        for j in range(0, int(len(polylines2[i])-1)):
            cv2.line(img,(int(polylines2[i][j][0]-pos_x),int(pos_y+shape_y-polylines2[i][j][1])),(int(polylines2[i][j+1][0]-pos_x),int(pos_y+shape_y-polylines2[i][j+1][1])),(0,0,1),thickness_lines[0])
    cv2.imwrite(b1,img)    


# ------------------------------------------------------------------------------------
# Develops massing calculations from drawn polylines
# ------------------------------------------------------------------------------------
def draw_land_use_analysis (polylines, line_type, folder, file_name):
    # generates and saves land use drawing
    image  = draw_paths (polylines, line_type, folder, file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # develops pixel count on drawing going through the list of land thickness
    land_use = []
    built_area = 0
    for i in range (0, len(color_canvas_rgb)):
        sought = color_canvas_rgb[i]
        result = np.count_nonzero(np.all(image==sought,axis=2)) * site_scale_factor * site_scale_factor * massing_height[i]
        land_use.append(result)
        built_area = built_area + result
    FAR = built_area/site_area()
    return land_use, built_area, FAR


# ------------------------------------------------------------------------------------
# Generates image with calculations which is served back to the feedback
# ------------------------------------------------------------------------------------
def report_land_use (data, file_name, session_folder, folder_name):
    if len(data[4]) > 1 :
        #brings data and generates drawing with land use analysis
        line_type = data[4]
        data.pop(4) # removes linetype array  
        data.pop(0) # removes style array (one element) from list
        ptexport=np.array(data).astype(int)  # turn into integer since mobile devices will produce fractions and pythonanywhere saves as float    
        line_type_export = np.array(line_type).astype(int)

        # saves line and stlyle data as numpy with base name as most up to date option both in the folder and the overall results dir
        file_path = os.path.join(session_folder, folder_name + '_massing.npy')
        np.save(file_path, ptexport)
        file_path = os.path.join(overall_results_directory, folder_name + '_massing.npy')
        np.save(file_path, ptexport)     
        file_path = os.path.join(session_folder, folder_name + '_massing_type.npy')
        np.save(file_path, line_type_export)
        file_path = os.path.join(overall_results_directory, folder_name + '_massing_type.npy')
        np.save(file_path, line_type_export)  

        points=ptexport.tolist() 
        polylines = pts_to_polylines (points, line_type) [0]
        linetype = pts_to_polylines (points, line_type) [1]
        name = file_name
        data_land_use = draw_land_use_analysis (polylines, linetype, session_folder, name)
    
        # generate new canvas
        canvas =Image.new('RGB',(canvas_x,canvas_y), color = 'white')
    
        #defines size of image pasted larger than other recommendation
        thumb_large_x = int(thumb_x * 1.5)
        thumb_large_y = int(thumb_y * 1.5)
        
        # place original image on the top
        im_source=draw_paths_base (polylines, linetype, session_folder,file_name)
        im_source = cv2.cvtColor(im_source, cv2.COLOR_BGR2RGB)
        im_source=cv2.rectangle(im_source,(0,0),(thumb_large_x,thumb_large_y),(0,0,0),3)
        im_source_pil = Image.fromarray(im_source)
        canvas.paste(im_source_pil,(0,30 + 56 + padding))
        
        # instantiates class for text. Uses larger text than other since information is lower
        sz1=80
        sz2=70
        font_large = ImageFont.truetype("arial.ttf", size = sz1)
        font_small = ImageFont.truetype("arial.ttf", size = sz2)
        draw = ImageDraw.Draw(canvas)
        
        # writes text
        draw.text((15, 30),"YOU DREW THIS...",(0,0,0),font=font_large)
        draw.text((15, thumb_large_y + padding+30),"WHICH MEANS..." ,(0,0,0),font=font_large) 
        draw.text((15, thumb_large_y + padding + 30 + sz1*1.5),'BUILT AREA = ' + str(int(data_land_use[1])) + 'm2',(0,0,0),font=font_small)    
        draw.text((15, thumb_large_y + padding + 30 + sz1*2*1.5),'FAR = ' + str(round(data_land_use[2],2)),(0,0,0),font=font_small)    

        # placwes image of original proposal
        im_source=cv2.imread(ucl_east_image)
        im_source = cv2.cvtColor(im_source, cv2.COLOR_BGR2RGB)
        im_source=cv2.rectangle(im_source,(0,0),(thumb_large_x,thumb_large_y),(0,0,0),3)
        im_source_pil = Image.fromarray(im_source)
        canvas.paste(im_source_pil,(int(canvas_y/2),30 + 56 + padding))

        # writes text about exsiting proposal
        draw.text((15 + int(canvas_x/2),30),"COMPARED TO...",(0,0,0),font=font_large)
        draw.text((15 + int(canvas_x/2), thumb_large_y + padding+30),"WHICH MEANS..." ,(0,0,0),font=font_large) 
        draw.text((15 + int(canvas_x/2), thumb_large_y + padding + 30 + sz1*1.5),'BUILT AREA = ' + str(ucl_east_development_area) + 'm2',(0,0,0),font=font_small)    
        draw.text((15 + int(canvas_x/2), thumb_large_y + padding + 30 + sz1*2*1.5),'STUDENTS = ' + str(ucl_east_student_population),(0,0,0),font=font_small)    

        
        # saves file
        b=os.path.join(session_folder,file_name +'_land_use_output'+'.jpg')
        canvas=canvas.resize((shape_x,shape_y), Image.ANTIALIAS)
        canvas.save(b)
    else:
        base=cv2.imread(link_base_image_warning)
        b=os.path.join(session_folder,file_name +'_land_use_output'+'.jpg')
        cv2.imwrite(b,base)


# ------------------------------------------------------------------------------------
# Reads sketch and extract abstract features from VGG19
# ------------------------------------------------------------------------------------
def feature_extract (sketch_to_analyze):
    # Read image file for network and process it to obtain abstract features for classification
    base_model = VGG19(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
    net = image.load_img((sketch_to_analyze), target_size=(224, 224))
    net = image.img_to_array(net)
    net = np.expand_dims(net, axis=0)
    net = preprocess_input(net)
    features = model.predict(net).flatten()
    return features


# ------------------------------------------------------------------------------------
# Sends sketch to feature_extract and uses the features to classify it 
# ------------------------------------------------------------------------------------
def style_classify (sketch_to_analyze):
    model_name=os.path.join(reference_directory, 'finalized_model.sav') 
    loaded_model = joblib.load(model_name)
    test_feature = feature_extract(sketch_to_analyze).reshape(1, -1)
    predicted_style=loaded_model.predict(test_feature)[0]
    return predicted_style


# ------------------------------------------------------------------------------------
# Sends sketch style_classify and uses data to generate style report served as image  
# ------------------------------------------------------------------------------------
def generate_neighbour_report (live_directory, file_name, declared_style):      
    # generate new canvas
    canvas =Image.new('RGB',(canvas_x,canvas_y), color = 'white')

    historic_styles_link=os.path.join(reference_directory, 'historic_styles' + '.npy')
    historic_styles=np.load(historic_styles_link)
    
    # place original image on the top
    sketch_to_analyze=os.path.join(live_directory, file_name + '.jpg')
    im_source=cv2.imread(sketch_to_analyze)
    im_source=cv2.rectangle(im_source,(0,0),(thumb_x,thumb_y),(0,0,0),3)
    im_source_pil = Image.fromarray(im_source)
    canvas.paste(im_source_pil,(0,0))

    # place base drawing beside the original
    basic_line_drawing=os.path.join(live_directory, file_name + '_base.jpg')
    im_source=cv2.imread(basic_line_drawing)
    im_source=cv2.rectangle(im_source,(0,0),(thumb_x,thumb_y),(0,0,0),3)
    im_source_pil = Image.fromarray(im_source)
    canvas.paste(im_source_pil,(padding+thumb_x,0))
    
    # develops style extraction analysis for sketch_to_analyze
    resulting_style = style_classify(sketch_to_analyze)
    
    # Instantiates class for text
    font_large = ImageFont.truetype("arial.ttf", size = 56)
    font_small = ImageFont.truetype("arial.ttf", size = 36)
    draw = ImageDraw.Draw(canvas)
    
    # writes text
    draw.text((15, 30),"YOU DREW THIS...",(0,0,0),font=font_large)        
    
    # draws delimiting lines
    draw.line((0, thumb_y+padding/2) + (canvas.size[0],thumb_y+padding/2), fill=0, width=10)
    draw.line((0, thumb_y*2+padding+padding/2) + (canvas.size[0],thumb_y*2+padding+padding/2), fill=0, width=10)
          
    #develops recommendations in case of success
    if resulting_style == declared_style:
        # place result icon beside
        style=historic_styles[resulting_style]
        result_icon_link=link_outcome_success
        im_source=cv2.imread(result_icon_link)
        im_source_pil = Image.fromarray(im_source)
        canvas.paste(im_source_pil,(2*(padding+thumb_x),0))

        #place three reference images
        for j in range (0,int(report_grid_x)):
            x_pos= (thumb_x + padding)*j
            y_pos= (thumb_y + padding)*(0+1)
            b=os.path.join(reference_directory_images,str(resulting_style)+'_'+str(j)+'.jpg')
            im_source=cv2.imread(b)
            im_source=cv2.cvtColor(im_source, cv2.COLOR_BGR2RGB)
            im_source_pil = Image.fromarray(im_source)
            canvas.paste(im_source_pil,(x_pos,y_pos))            
            
            #writes text
            draw.text((15, thumb_y*2 + padding*2 + 30),"...AND YOU GOT IT RIGHT!!!!",(0,0,0),font=font_large)
            draw.text((15, thumb_y*2 + padding*2 + 30 + 10 + 56),str(style),(0,0,0),font=font_small)            
        
    #develops recommendations in case of failure
    if resulting_style != declared_style:
        # place result icon beside
        style=historic_styles[resulting_style]
        result_icon_link=link_outcome_failure
        im_source=cv2.imread(result_icon_link)
        im_source_pil = Image.fromarray(im_source)
        canvas.paste(im_source_pil,(2*(padding+thumb_x),0))
                    
        #place three reference images to the side of each recommendation
        for j in range (0,int(report_grid_x)):
            x_pos= (thumb_x + padding)*j
            y_pos= (thumb_y + padding)*(0+1)
            b=os.path.join(reference_directory_images,str(resulting_style)+'_'+str(j)+'_sk.jpg')
            im_source=cv2.imread(b)
            im_source_pil = Image.fromarray(im_source)
            canvas.paste(im_source_pil,(x_pos,y_pos))         
        
        #writes text
        draw.text((15, thumb_y + padding+30),"...WHICH LOOKS..." + str(style) ,(0,0,0),font=font_large)
        
        # places correction below
        #place three reference images correction below
        style=historic_styles[declared_style]
        for j in range (0,int(report_grid_x)):
            x_pos= (thumb_x + padding)*j
            y_pos= (thumb_y + padding)*(1+1)
            b=os.path.join(reference_directory_images,str(declared_style)+'_'+str(j)+'_sk.jpg')
            im_source=cv2.imread(b)
            im_source_pil = Image.fromarray(im_source)
            canvas.paste(im_source_pil,(x_pos,y_pos))  
        
        #writes text
        draw.text((15, thumb_y*2 + padding*2 + 30),"...BUT WANTS TO LOOK...." + str(style),(0,0,0),font=font_large)

    # saves image and returns the path   
    b=os.path.join(live_directory,file_name +'_recommendation_output'+'.jpg')
    canvas=canvas.resize((shape_x,shape_y), Image.ANTIALIAS)
    canvas.save(b)


# ----------------------------------------------------------------------------------
# Calls for development of skeletons and basic lines
# ----------------------------------------------------------------------------------
def draw_skeleton_graphs (img, session_folder, file_name, folder_name):
    grey=skimage.img_as_ubyte(skimage.color.rgb2grey(img))
    binary=grey<250
    skel= morphology.skeletonize(binary)        
    skel=skel*1
    b4=os.path.join(session_folder, file_name + "_sk" + '.jpg')
    skel_draw=1-skel
    cv2.imwrite(b4,skel_draw) 
    #calls for development of dual graph and simplified lines bsaed on  skeleton
    sketch_grap=path_graph(skel,node_coords,threshold_distance,file_name,session_folder,folder_name, link_base_image,shape_x,shape_y)
    sketch_grap.draw_basic_connections()
    sketch_grap.draw_graph()


# ----------------------------------------------------------------------------------
# Generates images to be served as feedback from style and connectivty analysis
# ----------------------------------------------------------------------------------
def generate_image (ptexport, line_type, session_folder, file_name, folder_name, declared_style, task):
    points=ptexport.tolist()
    pols  = pts_to_polylines(points, line_type)[0]
    img=draw_paths (pols, line_type, session_folder,file_name) # draws paths in the small scale drawing ovwer white canvas for further processing
    draw_paths_base (pols, line_type, session_folder,file_name) # Draws paths in the small scale base
    draw_base_large(pols, session_folder,file_name)
    # will develop connectivity even for style to ensure final drawing considered at the end
    draw_skeleton_graphs(img, session_folder,file_name, folder_name)
    keras.backend.clear_session() # Needs to clean backend to perform more sessions
    # calls for feedbcak on style if task = 1 otherwise skips task if only connecitivty
    if task ==1: 
        generate_neighbour_report(session_folder,file_name,declared_style)


# ----------------------------------------------------------------------------------
# Develops grayscale blurred image of one single massing drawing for further aggregation
# ----------------------------------------------------------------------------------
def blurred_massing (filepath_np, filepath_linetype_np):
    # Read np data
    points_massing=np.load(filepath_np).tolist()
    line_type_massing=np.load(filepath_linetype_np).tolist()

    # generate polylines
    polylines = pts_to_polylines (points_massing, line_type_massing)[0]
    line_type = pts_to_polylines (points_massing, line_type_massing)[1]

    # draw polyline
    img = np.zeros((int(shape_x),int(shape_y),3), np.uint8)
    img.fill(255)
    for i in range(0, len(polylines)):
        thickness = thickness_lines[line_type[i]]
        color = color_canvas_rgb[line_type[i]]
        for j in range(0, int(len(polylines[i])-1)):
            cv2.line(img,(int(polylines[i][j][0]),int(shape_y-polylines[i][j][1])),(int(polylines[i][j+1][0]),int(shape_y-polylines[i][j+1][1])),color,thickness)

    # make greyscle and flur several times
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(0,3):
        img = cv2.GaussianBlur(img ,(195,195),cv2.BORDER_DEFAULT)

    # invert to make 255 = highest
    img = 255-img
    return img


# ----------------------------------------------------------------------------------
# Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it. Used for contour drawing generation
# ----------------------------------------------------------------------------------
def fig2data ( fig ):
    # http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( 700, 700, 4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


# ----------------------------------------------------------------------------------
# generates base massign averages contours with matplotlib still need checking!!!!
# ----------------------------------------------------------------------------------
def contour_based (list_drawings, list_drawings_types, colourmap):
    # develops imnitial image on top of which to add all blurred
    img = np.zeros((int(shape_x),int(shape_y),3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(int) # need to turn it into int otherwisde will not add more than 255
    
    # iterates through image list and adds all images to one single data to further remap
    for i in range (0, len(list_drawings)):
        filepath_np_massing = os.path.join(overall_results_directory, list_drawings[i])
        filepath_linetype_np_massing = os.path.join(overall_results_directory,list_drawings_types[i])
        img2 = blurred_massing (filepath_np_massing, filepath_linetype_np_massing)
        img2 = img2.astype(int)
        img = img + img2
    
    #remaps to 0 - 255 and turns itno integer
    img = img/(img.max()/255)
    img = img.astype(int)
    img=255-img
    img2=np.copy (img)
    img4=np.copy (img)
    img3 = color.gray2rgb(img2)
    # the remap operation messes up the types of integers in the array and this stuff is needed
    # https://stackoverflow.com/questions/40162890/opencv-cvtcolor-dtype-issueerror-215
    img=img.astype(np.float32) 
     
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    red_multiplier = [1,1, 0]
    img3 = red_multiplier * img3
    img3 = 255 - img3


    # generates matplotlib contour drawing out of the greyscale image
    fig = plt.figure(figsize=(7, 7), dpi=100, frameon=False)
    plt.autoscale(tight=True)
    plt.margins(0)
    x_vals = np.linspace(0, 700, 700)
    y_vals = np.linspace(0, 700, 700)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z=img4
    b=os.path.join(overall_results_directory,'massing_average1.jpg')

    plt.axis('off')
    cmap = plt.cm.get_cmap(colourmap)
    plt.contour(X, Y, Z, cmap=cmap)
    plt.savefig(b, bbox_inches="tight", pad_inches=0)

    # turns plt figure to np and make proper colors
    contour_data=fig2data(fig)   
    contour_data = np.flipud(contour_data) # we need to flip since matplotlib reads arrays downside up as opposed to np
    contour_data = cv2.cvtColor(contour_data, cv2.COLOR_BGR2RGB) # invers colours to make 3 channel and then warm colors
#    contour_data = cv2.cvtColor(contour_data, cv2.COLOR_RGB2BGR)

    # for some reason when saving data from fig the computer leaves an annoying white frame around.
    # This can be seein if using contourf (solid color)    
    # This is even after all possible efforts above telling the computer to remove frames and making things tight
    # needs cropping and resizing
    # If someone has a better idea, please go ahead 

    x1=88
    y1=87
    x2=629
    y2=615
    contour_cropped = contour_data[y1:y2, x1:x2]
    contour_cropped = cv2.resize(contour_cropped, dsize=(700, 700), interpolation=cv2.INTER_CUBIC)
    plt.imsave(b, contour_cropped)

    # brings in base image and overlays with contours
    base_image= cv2.imread(link_base_image)
    base_image=base_image.astype(np.float32)
    contour_cropped=contour_cropped.astype(np.float32)
    transparency = 0.5
    contour_base = cv2.addWeighted(contour_cropped,transparency,base_image,(1-transparency),0) 
#    
#    # returns image combined
    return contour_base

# ----------------------------------------------------------------------------------
# Generates analysis for all images saved so far
# ----------------------------------------------------------------------------------
def overall_image_report ():
    # reads files in directory and generates lists
    sketches=os.listdir(overall_results_directory)
    list_basic_connections=[]
    list_massing_drawings=[]
    list_massing_drawings_types=[]
    for i in sketches:          
        if not i == 'Thumbs.db':
            if re.findall('ln',i) :
                list_basic_connections.append(i)
            if re.findall('massing.npy',i):
                list_massing_drawings.append(i)
            if re.findall('massing_type.npy',i):
                list_massing_drawings_types.append(i)

    # ----------------------------------------------------------------------------------
    # generates line averages
    # ----------------------------------------------------------------------------------
    #loads basic line connections stored
    basic_connections=[]
    for i in list_basic_connections:
        filepath_np = os.path.join(overall_results_directory, i)
        connections = np.load(filepath_np).tolist()
        basic_connections.append(connections)
    
    #calls for the developmetnof combined simplified line drawing
    basic_line_drawing (node_coords, basic_connections, overall_results_directory, shape_x, shape_y,link_base_image)
    bundle_drawing (node_coords, basic_connections, overall_results_directory, shape_x, shape_y,link_base_image)
    
    # ----------------------------------------------------------------------------------
    # generates massing averages
    # ----------------------------------------------------------------------------------
    # calls function to generates contour base massign averages and saves file. Massing works with color: "autumn"
    contour_base_massing = contour_based (list_massing_drawings, list_massing_drawings_types, 'autumn')
    b=os.path.join(overall_results_directory,'massing_average.jpg')
    cv2.imwrite(b,contour_base_massing)

#%%
#overall_image_report ()









#%%




