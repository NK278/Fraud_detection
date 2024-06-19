from flask import render_template,jsonify,request,send_file,Flask
from src.exception import CustomException
from src.logger import logging as lg
from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredPipeline
import sys,os

app=Flask(__name__)


@app.route("/")
def home():
    return jsonify("Fraud Detection!")


@app.route("/train")
def train_route():
    try:
        train_pip=TrainingPipeline()
        train_pip.run_pip()
        return "Training Completed!"
    except Exception as e: raise CustomException(e,sys)


@app.route("/predict",methods=['GET','POST'])
def pred_route():
    try:
        if request.method == 'POST':
            prediction_pipeline = PredPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pip()

            lg.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prep_path,
                            download_name= prediction_file_detail.pred_file_name,
                            as_attachment= True)


        else:
            return render_template('upload_file.html')
    except Exception as e:  raise CustomException(e,sys)

if __name__ == "__main__":
    # app.run(debug=True)
    train_pip=TrainingPipeline()
    train_pip.run_pip()
    print("complete!")