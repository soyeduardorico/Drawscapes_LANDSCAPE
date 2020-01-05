from project_data import link_base_image, link_base_image_large_annotated, link_base_image_warning, ucl_east_image, link_feedback_massing_base
from project_data import shape_y, shape_x, thickness_lines, root_participation_directory
from project_data import reference_directory_images, reference_directory, overall_results_directory
from project_data import link_outcome_failure, link_outcome_success, node_coords, threshold_distance
from project_data import node_coords_large, color_canvas_rgb, node_coords_detailed
from project_data import site_scale_factor, massing_height
from project_data import ucl_east_development_area, ucl_east_student_population, ucl_east_research_area
from project_data import ratio_accomodation_base, ratio_accomodation_plinth, ratio_accomodation_tower, m2accomodation_per_student
from project_data import ratio_research_base, ratio_research_plinth, ratio_research_tower, databse_filepath
from project_data import feedback_barrier_base, feedback_canal_base, feedback_noise_base

#import os
import numpy as np
import cv2
import os
import tinydb
import skimage

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
import project_data as pdt

# ------------------------------------------------------------------------------------
# File location
# ------------------------------------------------------------------------------------
absFilePath = os.path.dirname(__file__)
root_data = os.path.join(absFilePath,  'data')
#%%

# ------------------------------------------------------------------------------------
# Testdevelopment for feedback on core detection
# ------------------------------------------------------------------------------------
#
import numpy as np
from scipy import ndimage as ndi
from feedback import generate_feedback_images
import skimage
from skimage import morphology
from skimage.morphology import watershed
from skimage.segmentation import random_walker
from skimage.feature import peak_local_max
from feedback import obtain_connections

#%%
#
user_id = '1577920980365'
#user_id='1578007367090'
#user_id='1578004791560'
millis = 1576064636649
file_name= user_id + '_2_'+ str(millis)

exercise = pdt.exercises[0]  #import massing data for this feedback operation
data_import = line_data_from_database(databse_filepath, user_id,exercise)
polylines = data_import[0]
linetype = data_import[1]



img= np.zeros((700,700,3), np.uint8)
img.fill(255)
#img = cv2.imread(pdt.link_base_image)

img=draw_paths_base (polylines, linetype, 'any', 'any', img, save='False')
img1 = cv2.imread(pdt.link_base_image)
img1=draw_paths_base (polylines, linetype, 'any', 'any', img1, save='False')
plt.imshow(img1)

text = obtain_connections(img)
print(text)

#
## draw massing and turn black and white
#img = np.zeros((int(shape_x),int(shape_y),3), np.uint8)
#img.fill(255)
#img=draw_paths_base (polylines_massing, linetype_massing, 'any', 'any', img, save='False')
#grey=skimage.img_as_ubyte(skimage.color.rgb2grey(img))
#binary=grey>250
#
## use distance_trasform to develop segmentation
#distance = ndi.distance_transform_edt(binary)
#plt.imshow(distance, cmap =plt.cm.gray)
#local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((2, 2)),
#                            labels=binary)
#
#markers = ndi.label(local_maxi)[0]
#print(markers)
#
#labels = watershed(-distance, markers, mask=binary,watershed_line=True)
#plt.imshow(labels, cmap=plt.cm.nipy_spectral)
#
#user_id_folder = os.path.join(root_data, user_id)
#temp_file_name = os.path.join(user_id_folder, 'temporal.jpg')
#
#cv2.imwrite(temp_file_name,labels)
#


#%%

#from tinydb import TinyDB, Query
#user_id='1577723064194'
#
#session_folder = os.path.join(pdt.root_participation_directory, user_id)
#user_db =os.path.join(session_folder,user_id + '_database.json')
#
#db = TinyDB(user_db)
#db.all()
#%%
#import time
#import datetime
#user_id='1576064564452'
#
#s = int(user_id) / 1000
#
#date_to_print = datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S')
#print(date_to_print )
