
//----------------------------------------------------------------------------------------------
// Common variables for all pages
//----------------------------------------------------------------------------------------------
var data_exercise = 1; //0 = draw over park, 1 = draw your own park
var data_canvas_x = 700; //size of canvas
var data_canvas_y = 700; //size of canvas
var color_canvas_scheme  = ['#000000','#e6c48a', '#ff6e5e','#ff0000','#2f5b7d', '#2c700f']; //pencil colors for paths (0), massing (1,2,3) and land use (4,5)  
var thickness_canvas_scheme =  [5,30,22,15, 10, 10];

//--------------------------------------------------------------------------
//initialises variables
//--------------------------------------------------------------------------  
var xcoords = new Array(); //dasta on position
var ycoords = new Array(); //data on position
var point = new Array(); //data on type of pressing for each point 1=push, 2 touchscreen, 3 microsoft pointer down, 4 mouse dowm 
var point_time = new Array(); //time at each poitn touch
var image_canvas = new Image();
var image_video = new Image();
var image_back = new Image();
var exportArray = new Array()
var style_save = new Array()
var linetype = new Array(); //data on type of line on the canvas. 0: black and 1,2,3 for buildigns

var color = new String ()
var timeStep = 1000;
var exportText = new String ();
var image_feedback_link = new String ();
var basename = new String ();
basename = "images/base_image.jpg"
basename2 = "images/video_1.gif"
var color1 = '#1d2952'
var color2 = '#1d2952'
var color3 = '#1d2952'
var color4 = '#1d2952'
var color5 = '#1d2952'
var color6 = '#1d2952'
var color7 = '#1d2952'
var color8 = '#1d2952'
var color9 = '#1d2952'
var color10 = '#ffffff'
var color_generic = '#000000'


//----------------------------------------------------------------------------------------------
// fucntion that sets all button colors to neutral
//----------------------------------------------------------------------------------------------
function color_buttons_neutral(){
    document.getElementById("btn1").style.background = color_generic;
    document.getElementById("btn2").style.background = color_generic;
    document.getElementById("btn3").style.background = color_generic;
    document.getElementById("btn4").style.background = color_generic;
    document.getElementById("btn5").style.background = color_generic;
    document.getElementById("btn6").style.background = color_generic;
    document.getElementById("btn7").style.background = color_generic;
    document.getElementById("btn8").style.background = color_generic;
    document.getElementById("btn9").style.background = color_generic;
    document.getElementById("btn10").style.background = color_generic;
}


//----------------------------------------------------------------------------------------------
// generates canvas with no background
//----------------------------------------------------------------------------------------------
function newWhiteCanvas(ln_type){
    ctx.strokeStyle = color_canvas_scheme[ln_type];
    ctx.lineWidth = thickness_canvas_scheme[ln_type];
    line_var=ln_type;   
    canvas.style.border = '1px solid black';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawMouse();
    drawTouch();
    drawPointer();
}


//----------------------------------------------------------------------------------------------
// Changes color and thickness of the stroke for the massig exercise
//----------------------------------------------------------------------------------------------
function newCanvasStroke(ln_type){
    ctx.strokeStyle = color_canvas_scheme[ln_type];
    ctx.lineWidth = thickness_canvas_scheme[ln_type];
    line_var=ln_type;
}


//----------------------------------------------------------------------------------------------
// Loads a new image on the background of the drawing canvas
//----------------------------------------------------------------------------------------------
function newcanvas(image, opacity, ln_type){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = opacity;
    ctx.drawImage(image,0,0);
    ctx.strokeStyle = color_canvas_scheme[ln_type];
    ctx.lineWidth = thickness_canvas_scheme[ln_type];
    line_var=ln_type;  
    ctx.lineCap = "round";
    canvas.style.border = '1px solid black';
}


//--------------------------------------------------------------------------
// Brings the button list below canvas
//--------------------------------------------------------------------------
function openTab(tabName) {
    var i, x;
    x = document.getElementsByClassName("containerTab");
    for (i = 0; i < x.length; i++) {
        x[i].style.display = "none";
    }
    document.getElementById(tabName).style.display = "block";
}


//----------------------------------------------------------------------------------------------
// Clears the canvas for new test drawing
//----------------------------------------------------------------------------------------------
function myClear(opacity, ln_type){
  newcanvas(image_canvas, opacity, ln_type);
  exportArray = [];
  xcoords = [];
  ycoords = [];
  point = []
  point_time=[];
  exportText=[];
}


//----------------------------------------------------------------------------------------------
// Waiting function
//----------------------------------------------------------------------------------------------
function wait(ms){
     var start = new Date().getTime();
     var end = start;
     while(end < start + ms) {
       end = new Date().getTime();
    }
}


//----------------------------------------------------------------------------------------------
// Collects data and passess on information as a single array
//----------------------------------------------------------------------------------------------
function mySave(declared_style){
    // writes drawing ionformation
    exportArray=[];
    style_save.push(declared_style);
    exportArray.push(style_save);
    exportArray.push(xcoords);
    exportArray.push(ycoords);// y information reversed
    exportArray.push(point);// information on type of touch / input
    exportArray.push(linetype);// information on linetype
    style_save=[]// removes style information after being used for further tests
    return exportArray;
}


//----------------------------------------------------------------------------------------------
// prototype to	start drawing on touch using canvas moveTo and lineTo
//----------------------------------------------------------------------------------------------
function drawTouch() {
	var start = function(e) {
		ctx.beginPath();
		x = e.changedTouches[0].pageX - this.offsetLeft;
		y = e.changedTouches[0].pageY- this.offsetTop;
		ctx.moveTo(x,y);
        xcoords.push(x);
        ycoords.push(y);
        tm = new Date().getTime();
        point_time.push(tm)
        point.push(1);
        linetype.push(line_var);
    };
    var move = function(e) {
      e.preventDefault();
      x = e.changedTouches[0].pageX - this.offsetLeft;
      y = e.changedTouches[0].pageY - this.offsetTop;
      ctx.lineTo(x,y);
      ctx.stroke();
      xcoords.push(x);
      ycoords.push(y);
      tm = new Date().getTime();
      point_time.push(tm)
      point.push(2);
      linetype.push(line_var);
  };
  document.getElementById("canvas").addEventListener("touchstart", start, false);
  document.getElementById("canvas").addEventListener("touchmove", move, false);
};


//----------------------------------------------------------------------------------------------
// prototype to	start drawing on pointer(microsoft ie) using canvas moveTo and lineTo
//----------------------------------------------------------------------------------------------
function drawPointer() {
	var start = function(e) {
        e = e.originalEvent;
        ctx.beginPath();
        x = e.pageX - this.offsetLeft;
        y = e.pageY - this.offsetTop;
        ctx.moveTo(x,y);
        xcoords.push(x);
        ycoords.push(y);
        var tm = new Date().getTime();
        t=tm-start_draw;
        point_time.push(t);
        point.push(1);
        linetype.push(line_var);        
    };
    var move = function(e) {
      e.preventDefault();
      e = e.originalEvent;
      x = e.pageX - this.offsetLeft;
      y = e.pageY - this.offsetTop;
      ctx.lineTo(x,y);
      ctx.stroke();
      xcoords.push(x);
      ycoords.push(y);
      var tm = new Date().getTime();
      t=tm-start_draw;
      point_time.push(t);
      point.push(3);
      linetype.push(line_var);
  };
  document.getElementById("canvas").addEventListener("MSPointerDown", start, false);
  document.getElementById("canvas").addEventListener("MSPointerMove", move, false);
}; 


//----------------------------------------------------------------------------------------------
// prototype to	start drawing on mouse using canvas moveTo and lineTo
//----------------------------------------------------------------------------------------------
function drawMouse () {
	var clicked = 0;
	var start = function(e) {
		clicked = 1;
		ctx.beginPath();
		x = e.pageX - this.offsetLeft;
		y = e.pageY- this.offsetTop;
		ctx.moveTo(x,y);
     xcoords.push(x);
     ycoords.push(y);
     tm = new Date().getTime();
     point_time.push(tm);
     point.push(1);
     linetype.push(line_var);
 };
 var move = function(e) {
  if(clicked){
     x = e.pageX - this.offsetLeft;
     y = e.pageY - this.offsetTop;
     ctx.lineTo(x,y);
     ctx.stroke();
     xcoords.push(x);
     ycoords.push(y);
     tm = new Date().getTime();
     point_time.push(tm);
     point.push(4);
     linetype.push(line_var);
 }
};
var stop = function(e) {
  clicked = 0;
};
document.getElementById("canvas").addEventListener("mousedown", start, false);
document.getElementById("canvas").addEventListener("mousemove", move, false);
document.addEventListener("mouseup", stop, false);
};




