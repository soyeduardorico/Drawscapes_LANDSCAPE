import math
import numpy as np
import pandas as pd
import cv2
import os
from datashader.bundling import hammer_bundle
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
        cv2.line(img,(node_coords[i[0]][0],node_coords[i[0]][1]),(node_coords[i[1]][0],node_coords[i[1]][1]),final_colour,
                 int(20*value/maxline))

    # Draws nodes
    for i in node_coords:
        cv2.circle(img,(i[0],i[1]),4,(0,0,0),-1)
        
    # Exports file
    filename=os.path.join(foldername + '\\basic_line_drawing.jpg')
    cv2.imwrite(filename,img)



#%% 
#node_coords=[[135,365],
#             [234,258],
#             [351,202],
#             [417,274],
#             [505,280],
#             [570,362],
#             [605,457],
#             [528,480],
#             [442,507],
#             [401,454]]
#
#edge_list=[[[5, 9], [0, 5], [0, 7], [3, 9]], [[2, 9], [3, 9], [0, 4], [5, 9]], [[0, 4], [1, 7], [3, 9], [1, 8]], [[3, 7], [5, 9], [1, 6], [1, 7]], [[0, 3], [3, 7], [7, 9], [0, 4]], [[5, 9], [2, 9], [4, 9], [2, 7]], [[0, 5], [4, 9], [4, 8], [5, 9]], [[1, 7], [0, 5], [0, 4], [7, 9]], [[0, 3], [4, 7], [1, 5], [1, 6]], [[3, 9], [0, 4], [3, 8], [0, 3]], [[3, 9], [3, 7], [4, 9], [1, 7]], [[4, 9], [5, 7], [1, 6], [0, 1]], [[0, 5], [3, 9], [4, 7], [2, 9]], [[0, 5], [3, 9], [4, 7], [0, 3]], [[0, 3], [0, 4], [0, 1], [4, 9]], [[3, 9], [1, 5], [0, 1], [0, 3]], [[0, 5], [3, 9], [0, 4], [0, 3]], [[3, 9], [5, 8], [1, 5], [0, 6]], [[0, 5], [1, 5], [0, 3], [4, 9]], [[3, 9], [4, 7], [1, 5], [0, 5]], [[5, 9], [1, 7], [5, 8], [0, 3]], [[0, 5], [3, 9], [0, 6], [7, 8]], [[4, 8], [3, 8], [3, 9], [0, 5]], [[1, 5], [5, 9], [3, 9], [0, 5]], [[2, 8], [5, 9], [3, 7], [2, 9]], [[3, 8], [0, 3], [5, 9], [4, 6]], [[0, 4], [2, 8], [0, 3], [2, 9]], [[4, 7], [3, 7], [0, 4], [4, 8]], [[0, 3], [5, 7], [5, 9], [2, 9]], [[0, 5], [0, 6], [0, 7], [1, 7]]]
#
#foldername='N:\\Documents\\PARTICIPATION\\INTERFACE INPUT\\DRAW_YOUR_PARK\\OVERALL_OUTPUT'
#link_base_image='N:\Documents\CODING\GUI\INTERFACE\FILES\DRAW YOUR PARK IMAGES\PRIMROSE HILL_RESIZE.jpg'
#shape_x=700
#
#shape_y=700
#
#    
#bundle_drawing(node_coords, edge_list, foldername, shape_x, shape_y,link_base_image)
#basic_line_drawing(node_coords, edge_list, foldername, shape_x, shape_y,link_base_image)
