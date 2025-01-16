import tkinter as tk
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import time

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (277, 277), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i , 2]
        if confidence > 0.8:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

class AppGui:

    def __init__(self, root):
        # window config
        self.root = root
        self.root.title("Detect Age & Gender App")
        self.root.geometry("600x300")

        # Scene management
        self.main_menu_frame = None
        self.models_menu_frame = None

        # button placeholders
        self.model_selection_button = None
        self.upload_predict_img_button = None
        self.live_detection_button = None
        self.pretrained_model_button = None
        self.load_custom_model_button = None

        self.current_image_path = None
        self.current_model_path = None
        self.model = None
        self.gender_dict = {0:'Male', 1:'Female'}

        # flag
        self.isPretrainedModelUsed = False
        self.isLoadedModelTextSet = False

        # pretrained models
        self.faceProto = "pretrained_models/opencv_face_detector.pbtxt"
        self.faceModel = "pretrained_models/opencv_face_detector_uint8.pb"

        self.ageProto = "pretrained_models/age_deploy.prototxt"
        self.ageModel = "pretrained_models/age_net.caffemodel"

        self.genderProto = "pretrained_models/gender_deploy.prototxt"
        self.genderModel = "pretrained_models/gender_net.caffemodel"

        self.faceNet = cv2.dnn.readNet(self.faceModel, self.faceProto)
        self.ageNet = cv2.dnn.readNet(self.ageModel,self.ageProto)
        self.genderNet = cv2.dnn.readNet(self.genderModel, self.genderProto)

        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746) 

        self.mainMenuScene()

    def mainMenuScene(self):
        self.clearFrame()

        self.main_menu_frame = tk.Frame(self.root)
        self.main_menu_frame.pack()

        self.title_label = tk.Label(
            self.main_menu_frame, text="Detect Age & Gender", font=("Arial", 16)
        )
        self.title_label.pack(pady=10)

        self.model_selection_button = ttk.Button(
            self.main_menu_frame, text="Select Prediction Model", command=self.selectModel
        )
        self.model_selection_button.pack(pady=10, ipadx=20, ipady=10)

        self.buttonFrame = tk.Frame(self.main_menu_frame)
        self.buttonFrame.pack(pady=10)

        self.upload_predict_img_button = ttk.Button(
            self.buttonFrame, text="Import Image", command=self.uploadImage
        )
        self.upload_predict_img_button.grid(row=0, column=0, padx=10, ipadx=20, ipady=10)

        self.live_detection_button = ttk.Button(
            self.buttonFrame, text="Live Detection", command=self.liveDetection
        )
        self.live_detection_button.grid(row=0, column=1, padx=10, ipadx=20, ipady=10)

        if self.current_image_path:
            self.image_label = tk.Label(
                self.main_menu_frame, text=f"Loaded Image: {os.path.basename(self.current_image_path)}"
            )
            self.image_label.pack(pady=5)

            self.predict_button = ttk.Button(
                self.main_menu_frame, text="Predict Age & Gender", command=self.predictAgeGender
            )
            self.predict_button.pack(pady=10, ipadx=20, ipady=10)

    def selectModelsMenu(self):
        self.clearFrame()

        self.models_menu_frame = tk.Frame(self.root)
        self.models_menu_frame.pack(fill=tk.BOTH, expand=True)

        self.title_label = tk.Label(
            self.models_menu_frame, text="Detect Age & Gender", font=("Arial", 16)
        )
        self.title_label.pack(pady=10)

        self.pretrained_model_button = ttk.Button(
            self.models_menu_frame, text="Load Pretrained Model", command=self.loadPretrainedModel
        )
        self.pretrained_model_button.pack(pady=10, ipadx=10, ipady=5)

        self.load_custom_model_button = ttk.Button(
            self.models_menu_frame, text="Load Custom Model", command=self.loadCustomModel
        )
        self.load_custom_model_button.pack(pady=10, ipadx=10, ipady=5)

        # If a model was previously loaded, recreate the label
        if self.isLoadedModelTextSet and self.isPretrainedModelUsed:
            self.model_label = tk.Label(
                self.models_menu_frame, text=f"Loaded Model: Caffe model"
            )
            self.model_label.pack(pady=5)
        if self.isLoadedModelTextSet and self.isPretrainedModelUsed == False:
            self.model_label = tk.Label(
                self.models_menu_frame, text=f"Loaded Model: {os.path.basename(self.current_model_path)}"
            )
            self.model_label.pack(pady=5)

        self.back_button = ttk.Button(
            self.models_menu_frame, text="Back", command=self.mainMenuScene
        )
        self.back_button.place(relx=0.01, rely=0.95, anchor="sw")

    def clearFrame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def selectModel(self):
        print("Select prediction model")
        self.selectModelsMenu()

    def liveDetection(self):
        genderList = ['Male', 'Female']

        # open laptop camera
        video = cv2.VideoCapture(0)

        padding = 20
        while True:
            ret, frame=video.read()
            frame, bboxs = faceBox(self.faceNet, frame)
            for bbox in bboxs:
                #face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] # x-y width height
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1), max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                self.genderNet.setInput(blob) #pass blob into the gender net
                genderPrediction = self.genderNet.forward() #extract the prediction
                gender = genderList[genderPrediction[0].argmax()] #max prediction value (best prediction)


                self.ageNet.setInput(blob) #pass blob into age net
                agePrediction = self.ageNet.forward() #extract prediction
                age = self.ageList[agePrediction[0].argmax()]

                label = "{}, {}".format(gender, age)
                cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Age-Gender", frame)
            k=cv2.waitKey(1)
            if k == ord('q'):
                break
            
        video.release()
        cv2.destroyAllWindows()

    def uploadImage(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
        )
    
        if file_path:
            self.current_image_path = file_path
            print(f"Selected Image: {self.current_image_path}")
        
            if hasattr(self, "image_label"):
                self.image_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            else:
                self.image_label = tk.Label(self.main_menu_frame, text=f"Loaded Image: {os.path.basename(file_path)}")
                self.image_label.pack(pady=5)

            if hasattr(self, "predict_button"):
                self.predict_button.pack_forget()

            self.predict_button = ttk.Button(
                self.main_menu_frame, text="Predict Age & Gender", command=self.predictAgeGender
            )
            self.predict_button.pack(pady=10, ipadx=20, ipady=10)

    def predictAgeGender(self):
        if self.model == None and self.isPretrainedModelUsed == False:
            print("Error! Model not loaded. Load a model first!")
        elif self.isPretrainedModelUsed:
            self.usePretraintedModel()
            print("use pretrained Models")
        else:
            processed_img = self.loadAndPreprocessImage(self.current_image_path)
            predictions = self.model.predict(processed_img.reshape(1, 128, 128, 1))

            predicted_gender = self.gender_dict[round(predictions[0][0][0])]
            predicted_age = round(predictions[1][0][0])
            original_img = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale image for display

            plt.imshow(original_img, cmap='gray')  # Show in grayscale
            plt.axis('off')
            plt.title(f"Predicted Gender: {predicted_gender}\nPredicted Age: {predicted_age:.1f} years")
            plt.show()

    def loadPretrainedModel(self):
        self.isPretrainedModelUsed = True
        self.isLoadedModelTextSet = True
        if hasattr(self, "model_label"):
            self.model_label.config(text=f"Loaded Model: Caffe model")
        else:
            self.model_label = tk.Label(self.models_menu_frame, text=f"Loaded Model: Caffe model")
            self.model_label.pack(pady=5)

    def loadCustomModel(self):
        self.isPretrainedModelUsed = False
        self.isLoadedModelTextSet = True

        file_path = filedialog.askopenfilename(
            title="Select a Model File",
            filetypes=[("Model Files", "*.h5;*.tflite;*.onnx;*.pb;*.pt;*.pkl")]
        )

        if file_path:
            self.current_model_path = file_path
            self.loadAndCompileModel(self.current_model_path)
            print(f"Selected Model: {self.current_model_path}")

            if hasattr(self, "model_label"):
                self.model_label.config(text=f"Loaded Model: {os.path.basename(file_path)}")
            else:
                self.model_label = tk.Label(self.models_menu_frame, text=f"Loaded Model: {os.path.basename(file_path)}")
                self.model_label.pack(pady=5)

    def loadAndCompileModel(self, model_path):
        #load
        self.model = tf.keras.models.load_model(model_path, compile=False)
        #compile
        self.model.compile(
            loss=[tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.MeanAbsoluteError()], 
            optimizer='adam', 
            metrics=['accuracy']
        )

    def loadAndPreprocessImage(self, image_path, target_size=(128, 128)):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
        img = cv2.resize(img, target_size)  # Resize image
        img = img.astype('float32') / 255.0  # Normalize pixel values to [0,1]
        img = np.expand_dims(img, axis=-1)  # Add channel dimension (H, W, 1)
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, H, W, 1)
        return img

    def usePretraintedModel(self):
        img = cv2.imread(self.current_image_path)
        img = cv2.resize(img, (720, 640))
        genderList = ['Male', 'Female']

        image_cp = img.copy()
        img_h = image_cp.shape[0]
        img_w = image_cp.shape[1]
        blob = cv2.dnn.blobFromImage(image_cp, 1.0, (300, 300), self.MODEL_MEAN_VALUES, True, False)

        self.faceNet.setInput(blob)
        detected_faces = self.faceNet.forward()

        face_bounds = []

        for i in range(detected_faces.shape[2]):
            confidence = detected_faces[0, 0, i, 2]
            #first that has the higher confidence
            if confidence > 0.99:
                #Map a rectangle on the image -face
                x1 = int(detected_faces[0, 0, i, 3] * img_w)
                y1 = int(detected_faces[0, 0, i, 4] * img_h)
                x2 = int(detected_faces[0, 0, i, 5] * img_w)
                y2 = int(detected_faces[0, 0, i, 6] * img_h)
                cv2.rectangle(image_cp, (x1, y1), (x2, y2), (0, 255, 0), int(round(img_h/150)), 8)
                face_bounds.append([x1, y1, x2, y2])

        if not face_bounds:
            print("No faces were detected!")
            exit

        for face_bound in face_bounds:
            try:
                #exctract the relevant data coresponding to the face
                face = image_cp[max(0, face_bound[1] - 15): min(face_bound[3] + 15, image_cp.shape[0] - 1), #y axes bounds
                                max(0, face_bound[0] - 15): min(face_bound[2] + 15, image_cp.shape[1] - 1)] #x axes bounds
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, True)
                self.genderNet.setInput(blob)
                gender_prediction = self.genderNet.forward()
                gender = genderList[gender_prediction[0].argmax()]  #Example [[9.9915099e-01 8.4903854e-04]] 99% that is a man and 84% that is a female

                self.ageNet.setInput(blob)
                age_prediction = self.ageNet.forward()
                age_result = self.ageList[age_prediction[0].argmax()]

                cv2.putText(image_cp, f'{gender}, {age_result}', (face_bound[0], face_bound[1] + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 4, cv2.LINE_AA)
            except Exception as e:
                print(e)
                continue
            
        cv2.imshow('Result', image_cp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    root = tk.Tk()
    app = AppGui(root)

    root.mainloop()
