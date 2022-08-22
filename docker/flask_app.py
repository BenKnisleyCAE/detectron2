"""
CAE Detectron Docker Flask API
"""
import os
import cv2

## import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

## Import Flask, and its extras
from flask import Flask, request

## Create and configure a Detectron configuration object
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # Set threshold for this model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/appuser/detectron2_repo/model.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

## Create a predictor object, using the configuration object
predictor = DefaultPredictor(cfg)

## Create a Flask application instance
app = Flask(__name__)

## Create a invalid file exception
class InvalidInputFile(Exception):
    def __init__(self, msg):
        self.message = msg

## Define predict function
def predict(input_path, output_path):
    """
    Runs the trained Detectron model on the given input file.

    Parameters:
        - input_path (str): The path of the image file to run the model on.

        - output_path (str): The path of the output image.
    
    Returns:
        None
    """
    ## Read image with OpenCV
    im = cv2.imread(input_path)

    ## Image valid checkpoint
    if im is None:
        raise InvalidInputFile("Given input file does not exist, or is not an image file.")

    ## Generate prediction image
    outputs = predictor(im)  
    
    ## Draw output predictions over image
    v = Visualizer(im[:, :, ::-1])
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    ## Save image to file
    cv2.imwrite(output_path, out.get_image())

## Define welcome route 
@app.route('/')
def hello_route():
    """
    This route serves a welcome message to the caller. It provides nothing, and 
    should be used only to see if the API is running.
    """
    return "Welcome to CAE Detectron Predict API", 200

## Define predict route
@app.route('/predict',  methods=['POST'])
def predict_route():
    """
    Flask route to run the Detectron 
    """
    ## Get dict object containing form fields   
    form_dict = request.get_json() if request.is_json else request.form

    ## Extract values from form_dict obj
    input_dataset_name = form_dict.get('input_dataset')
    output_dataset_name = form_dict.get('output_dataset')

    ## Parameter check-point to make sure they were given and not none 
    if not all([input_dataset_name, output_dataset_name]):
        return 'Invalid parameters', 400

    ## Join dataset names to i/o paths
    input_path = os.path.join('/mnt/input', input_dataset_name)
    output_path = os.path.join('/mnt/output', output_dataset_name)

    ## Call predict function
    predict(input_path, output_path)

    ## Return success message
    return output_path, 202

## Run flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7008)