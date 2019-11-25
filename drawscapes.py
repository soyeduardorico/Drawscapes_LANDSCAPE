# ----------------------------------------------------------------------------------
# For running flask on pythonanywhere  see: https://www.youtube.com/watch?v=M-QRwEEZ9-8 
# For session generation see https://www.youtube.com/watch?v=T1ZVyY1LWOg
# For general tutorials on flask: https://www.youtube.com/watch?v=MwZwr5Tvyxo
# ----------------------------------------------------------------------------------
from flask import Flask, render_template, url_for, request, jsonify, send_from_directory, session, redirect, make_response, flash
import time
import os

from drawing_app_functions import drawscapes_feedback_function, report_land_use, drawscapes_draw_base, drawscapes_draw_base_2, save_land_uses 
from style_transfer import call_montage


# ----------------------------------------------------------------------------------
# instantiates app and generates cookie to be passed to client
# ----------------------------------------------------------------------------------
app=Flask(__name__)
app.secret_key = os.urandom(24)
absFilePath = os.path.dirname(__file__)
root_data = os.path.join(absFilePath,  'data')


# ----------------------------------------------------------------------------------
# initiates Redis Queue when running in Ubuntu. Comment when running tests on Windows
# ----------------------------------------------------------------------------------
# import redis
# from rq import Queue
# r=redis.Redis()
# q=Queue(connection=r)


# -----------------------------------------------------------------------------------------
# renders index page
# -----------------------------------------------------------------------------------------
@app.route ('/index')
def index():
    millis = int(round(time.time() * 1000))
    variable = str(millis)
    session['user'] = variable
    session_folder=os.path.join(root_data, variable)
    os.mkdir(session_folder)
    return render_template ('index.html')


# -----------------------------------------------------------------------------------------
# closes session
# -----------------------------------------------------------------------------------------
@app.route('/dropsession')
def dropsession():
    session.pop('user', None)
    return 'Dropped'


# -----------------------------------------------------------------------------------------
# fabricates a url for non static folder a seen in https://www.youtube.com/watch?v=Y2fMCxLz6wM
# -----------------------------------------------------------------------------------------
@app.route('/data/<filename>')
def data(filename):
    # use the line below to work with cookies in the browser as opposed to sessions
    # session = request.cookies.get("session_number")
    target_directory = 'data/' + session['user'] 
    return send_from_directory(target_directory, filename)

@app.route('/overall_results/<filename>')
def overall_results(filename):
    # use the line below to work with cookies in the browser as opposed to sessions
    # session = request.cookies.get("session_number")
    target_directory = 'overall_results/'
    return send_from_directory(target_directory, filename)


# -----------------------------------------------------------------------------------------
# Routes to different web pages
# -----------------------------------------------------------------------------------------
@app.route('/drawscapes')
def drawscapes():
    # defines session number and generates folder for further saving files 
    # passes session number to browser as cookie 'session_number' 
    # res = make_response("Setting a cookie")
    # res.set_cookie('session_number', variable, max_age=60*60*24*365*2)
    return render_template ('drawscapes.html', 
        title = 'network design for session ' + session['user'])

@app.route('/drawscapes_intro')
def drawscapes_intro():
    return render_template ('drawscapes_intro.html', 
        title = 'network design for session ' + session['user'])

@app.route('/drawscapes_massing/<filename>')
def drawscapes_massing(filename):
    return render_template ('drawscapes_massing.html', 
        title = 'massing design for session ' + session['user'], 
        imagename=filename)

@app.route('/drawscapes_land_use/<filename>')
def drawscapes_land_use(filename):
    return render_template ('drawscapes_land_use.html', 
        title = 'massing design for session ' + session['user'], 
        imagename=filename)

@app.route('/drawscapes_conclussion')
def drawscapes_conclussion():
    return render_template ('drawscapes_conclussion.html', title = session['user'])

@app.route('/drawscapes_form')
def drawscapes_form():
    return render_template ('drawscapes_form.html', title = session['user'])


# -----------------------------------------------------------------------------------------
# Routes to functions feeding back info into front end
# -----------------------------------------------------------------------------------------
@app.route('/drawscapes_feedback', methods=["GET", "POST"])
def drawscapes_feedback():
    # use the line below to work with cookies in the browser as opposed to sessions
    # session = request.cookies.get("session_number")
 
    # defines drawing number within the session
    millis = int(round(time.time() * 1000))
    session_folder=os.path.join(root_data, session['user']) # uses same folder as folder session
    file_name= session['user']+'_'+ str(millis)
    folder_name=session['user']
    task = 1 # 1 for style, 2 for connectivity. This will tell feedack function to not carry all tasks

    # ----------------------------------------------------------------------------------
    # brings json data and calls drawing feedback into the queue. Activate on Ubuntu
    # ----------------------------------------------------------------------------------
    # data = request.json
    # job = q.enqueue(drawscapes_feedback_function, data, file_name, session_folder, folder_name, task)
    # while job.is_finished != True: 
    #     time.sleep(0.1)
    
    # ----------------------------------------------------------------------------------
    # brings json data and calls drawing feedback. Activate on Windows
    # ----------------------------------------------------------------------------------
    data = request.json
    drawscapes_feedback_function(data, file_name, session_folder, folder_name, task)

    # sends name of file back to browswer
    image_feedback=  file_name + '_recommendation_output.jpg' # defines name of image for feedbak and passes it to template
    return jsonify(image_feedback)
   
@app.route('/drawscapes_connection_feedback', methods=["GET", "POST"])
def drawscapes_connection_feedback():
    # use the line below to work with cookies in the browser as opposed to sessions
    # session = request.cookies.get("session_number")
    # defines drawing number within the session
    millis = int(round(time.time() * 1000))
    session_folder=os.path.join(root_data,  str(session['user'])) # uses same folder as folder session
    file_name = str(session['user']) + '_' +str(millis)
    folder_name=session['user']
    task = 2 # 1 for style, 2 for connectivity. This will tell feedack function to not carry all tasks 

    # ----------------------------------------------------------------------------------
    # brings json data and calls drawing feedback into the queue. Activate on Ubuntu
    # ----------------------------------------------------------------------------------
    # data = request.json
    # job = q.enqueue(drawscapes_feedback_function, data, file_name, session_folder, folder_name, task)
    # while job.is_finished != True:
    #      time.sleep(0.1)
    
    # ----------------------------------------------------------------------------------
    # brings json data and calls drawing feedback. Activate on Windows
    # ----------------------------------------------------------------------------------
    data = request.json
    drawscapes_feedback_function(data, file_name, session_folder, folder_name, task)

    # sends name of file back to browswer        
    image_feedback=  file_name + '_ln.jpg' # defines name of image for feedbak and passes it to template
    return jsonify(image_feedback)

@app.route('/drawscapes_massing_base', methods=["GET", "POST"])
def drawscapes_massing_base():
    # defines drawing number within the session
    millis = int(round(time.time() * 1000))
    session_folder=os.path.join(root_data, session['user']) # uses same folder as folder session
    file_name= session['user']+'_'+ str(millis)
    folder_name=session['user']

    # ----------------------------------------------------------------------------------
    # brings json data and calls drawing feedback into the queue. Activate on Ubuntu
    # ----------------------------------------------------------------------------------
    # data = request.json
    # job = q.enqueue(drawscapes_draw_base, data, file_name, session_folder, folder_name)
    # while job.is_finished != True:
    #     time.sleep(0.1)
    
    # ----------------------------------------------------------------------------------
    # brings json data and calls for development of image style input to the canvas. Activate on Windows
    # ----------------------------------------------------------------------------------
    data = request.json
    drawscapes_draw_base (data, file_name, session_folder, folder_name) # Draws paths in the small scale base

    # sends name of file back to browswer
    image_feedback=  file_name + '_base.jpg' # defines name of image for feedbak and passes it to template
    return jsonify(image_feedback)

@app.route('/drawscapes_landscape_base', methods=["GET", "POST"])
def drawscapes_landscape_base():
    # defines drawing number within the session
    millis = int(round(time.time() * 1000))
    session_folder=os.path.join(root_data, session['user']) # uses same folder as folder session
    file_name= session['user']+'_'+ str(millis)
    folder_name=session['user']
    
    # ----------------------------------------------------------------------------------
    # brings json data and calls drawing feedback into the queue. Activate on Ubuntu
    # ----------------------------------------------------------------------------------
    # data = request.json
    # job = q.enqueue(drawscapes_draw_base_2, data, file_name, session_folder, folder_name)
    # while job.is_finished != True: 
    #     time.sleep(0.1)
    
    # ----------------------------------------------------------------------------------
    # brings json data and calls for development of image style input to the canvas. Activate on Windows
    # ----------------------------------------------------------------------------------
    data = request.json
    drawscapes_draw_base_2 (data, file_name, session_folder, folder_name)  

    # sends name of file back to browswer
    image_feedback=  file_name + '_landscape_base.jpg' # defines name of image for feedbak and passes it to template
    return jsonify(image_feedback)

@app.route('/drawscapes_save_land_uses', methods=["GET", "POST"]) #complete when generating landscape
def drawscapes_save_land_uses():
    # defines drawing number within the session
    millis = int(round(time.time() * 1000))
    session_folder=os.path.join(root_data, session['user']) # uses same folder as folder session
    file_name= session['user']+'_'+ str(millis)
    folder_name = session['user'] # Used as base for drawscapes_massing.html canvas base 

    # ----------------------------------------------------------------------------------
    # brings json data and calls drawing feedback into the queue. Activate on Ubuntu
    # ----------------------------------------------------------------------------------
    # data = request.json
    # job = q.enqueue(save_land_uses, data, session_folder, file_name, folder_name)
    # while job.is_finished != True: 
    #     time.sleep(0.1)
    
    # ----------------------------------------------------------------------------------
    # brings json data and calls for development of image style input to the canvas. Activate on Windows
    # ----------------------------------------------------------------------------------
    data = request.json
    save_land_uses (data, session_folder, file_name, folder_name)

    # sends name of file back to browswer
    image_feedback=  folder_name + '_combined.jpg' # defines name of image for feedbak and passes it to template
    return jsonify(folder_name)

@app.route('/drawscapes_land_use_analysis', methods=["GET", "POST"])
def drawscapes_land_use_analysis():
    # defines drawing number within the session
    millis = int(round(time.time() * 1000))
    session_folder=os.path.join(root_data, session['user']) # uses same folder as folder session
    file_name= session['user']+'_'+ str(millis)
    folder_name=session['user']
    
    # ----------------------------------------------------------------------------------
    # brings json data and calls drawing feedback into the queue. Activate on Ubuntu
    # ----------------------------------------------------------------------------------
    # data = request.json
    # job = q.enqueue(report_land_use, data, file_name, session_folder, folder_name)
    # while job.is_finished != True: 
    #     time.sleep(0.1)
    
    # ----------------------------------------------------------------------------------
    # brings json data and calls drawing feedback. Activate on Windows
    # ----------------------------------------------------------------------------------
    data = request.json
    report_land_use (data, file_name, session_folder, folder_name)

    # sends name of file back to browswer
    image_feedback=  file_name + '_land_use_output.jpg' # defines name of image for feedbak and passes it to template
    return jsonify(image_feedback)   

@app.route('/drawscapes_save_text', methods=["GET", "POST"])
def drawscapes_save_text():
    session_folder=os.path.join(root_data, session['user']) # uses same folder as folder session
    file_name= session['user']
    file_path = os.path.join(session_folder, file_name + '.txt')

    # ----------------------------------------------------------------------------------
    # brings json data and calls drawing feedback.
    # ----------------------------------------------------------------------------------
    data = request.json
    data=str(data) # just in case
    textfile = open(file_path, 'w')
    textfile.write(data)
    textfile.close()
    dropsession()   


if __name__ == "__main__":
    millis=0
    points = []
    number_iterations  = 1
    # serve(app, host='0.0.0.0', port=80)
    app.run(debug=True, threaded=True)# requires threads to run parallel requests independetly 

  
