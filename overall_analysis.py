import math
import numpy as np
import pandas as pd
import cv2
import os
# from datashader.bundling import hammer_bundle
import networkx as nx
import matplotlib



# ------------------------------------------------------------------------------------
# Defines bundle drawing from a series of existing basic line drawings
# ------------------------------------------------------------------------------------
def bundle_drawing (node_coords, edge_list, foldername, shape_x, shape_y,link_base_image):
    # Reads the list of nodes and turns it into a PD dataframe for further analysis       
    n=len(node_coords)
    nodes = pd.DataFrame(["node"+str(i) for i in range(n)], columns=['name'])
    nodes.tail()
    xcol=[]
    ycol=[]    
    for i in node_coords:
        xcol.append(i[0])
        ycol.append(i[1])    
    nodes['x'] = pd.DataFrame(xcol)
    nodes['y'] = pd.DataFrame(ycol)
    nodes.tail()
    
    # Reads the list of edges found by the script and turns it into a PD dataframe for further analysis       
    sourcecol=[]
    targetcol=[]
    for i in edge_list:
        for j in i:
            sourcecol.append(j[0])
            targetcol.append(j[1])
    sourcecol=np.array(sourcecol)
    targetcol=np.array(targetcol)
    edges=np.vstack((sourcecol,targetcol))
    edges=np.transpose(edges)
    edges = pd.DataFrame(edges,columns=['source', 'target'])   
    
    # Generate bundle drawing         
    BC=hammer_bundle(nodes,edges,initial_bandwidth=0.15)
    
    # Generatebundle drawing from BC 
    img = cv2.imread(link_base_image)
    for i in range (len(BC.values)):
        if math.isnan(BC.values[i][0]) == False:
            if math.isnan(BC.values[i+1][0]) == False:
                cv2.line(img,(int(BC.values[i][0]),int(BC.values[i][1])),(int(BC.values[i+1][0]),int(BC.values[i+1][1])),(0,0,0),4)

    # Draws nodes
    for i in node_coords:
        cv2.circle(img,(i[0],i[1]),5,(0,0,255),-1)

    # Exports file
    filename=os.path.join(foldername + '\\bundle_drawing.jpg')
    cv2.imwrite(filename,img)


# ------------------------------------------------------------------------------------
# Defines colourcoded line drawing from a series of existing basic line drawings
# ------------------------------------------------------------------------------------
def basic_line_drawing (node_coords, edge_list, foldername,shape_x,shape_y,link_base_image):
    # Goes through edges, generates a graph and gives edges a larger value igf used several times    
    graph=nx.Graph()
    maxline=1
    for i in edge_list:
        for j in i:
            e=(j[0],j[1])
            if graph.has_edge(*e):            
                w = graph[j[0]][j[1]]['weight']+1
                graph.add_edge(j[0],j[1],weight=w)
                if w > maxline:
                    maxline = w
            else:
                graph.add_edge(j[0],j[1],weight=1)
    
    
    # Defines colour code for order drawing and draws lines
    cmap = matplotlib.cm.get_cmap('brg')
    img = cv2.imread(link_base_image)    
    for i in graph.edges:
        value=graph[i[0]][i[1]]['weight']
        colourvalue = value/maxline
        rgba = cmap(0.5-colourvalue/2)
        final_colour= (int(rgba[0]*255),int(rgba[1]*255),int(rgba[2]*255)) 
        if int(20*value/maxline)>0:
            cv2.line(img,(node_coords[i[0]][0],node_coords[i[0]][1]),(node_coords[i[1]][0],node_coords[i[1]][1]),final_colour,
                 int(20*value/maxline))

    # Draws nodes
    for i in node_coords:
        cv2.circle(img,(i[0],i[1]),4,(0,0,0),-1)
        
    # Exports file
    filename=os.path.join(foldername + '\\basic_line_drawing.jpg')
    cv2.imwrite(filename,img)



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


# ------------------------------------------------------------------------------------
# Generates conclussion drawings for all categories (_lines, _massing and _land_uses)
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
