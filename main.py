import cv2 # type: ignore
from tqdm.notebook import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


new_model = tf.keras.models.load_model('age_gender_model.h5', compile=False)

new_model.compile(
    loss=[tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.MeanAbsoluteError()], 
    optimizer='adam', 
    metrics=['accuracy']
)
gender_dict = {0:'Male', 1:'Female'}
# image = 'face_images/woman.jpg'
# img = tf.keras.preprocessing.image.load_img(image, color_mode='grayscale')
# #predict from model
# pred = new_model.predict(img.reshape(1, 128, 128, 1))
# pred_gender = gender_dict[round(pred[0][0][0])]
# pred_age = round(pred[1][0][0])
# print("Predicted Gender:",pred_gender, "Predicted Age:", pred_age)
# plt.axis('off')
# plt.imshow(img.reshape(128, 128), cmap='gray');

# new_model.summary()


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Loads an image, converts to grayscale, resizes, normalizes, and reshapes it for model input."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
    img = cv2.resize(img, target_size)  # Resize image
    img = img.astype('float32') / 255.0  # Normalize pixel values to [0,1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (H, W, 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, H, W, 1)
    return img


image_path = 'face_images/old_lady.jpg'  # Change to your image path
processed_img = load_and_preprocess_image(image_path)

# predictions = new_model.predict(processed_img)
# predicted_gender = 'Male' if predictions[0][0] > 0.5 else 'Female'
# predicted_age = float(predictions[1][0])  # Age is a regression output
predictions = new_model.predict(processed_img.reshape(1, 128, 128, 1))

predicted_gender = gender_dict[round(predictions[0][0][0])]
predicted_age = round(predictions[1][0][0])
print("Gender prediction: ", predictions[0][0][0])


original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale image for display

plt.imshow(original_img, cmap='gray')  # Show in grayscale
plt.axis('off')
plt.title(f"Predicted Gender: {predicted_gender}\nPredicted Age: {predicted_age:.1f} years")
plt.show()




def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (277, 277), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i , 2]
        if confidence > 0.99:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs
    

faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# open laptop camera
video = cv2.VideoCapture(0)

padding = 20
while True:
    ret, frame=video.read()
    frame, bboxs = faceBox(faceNet, frame)
    for bbox in bboxs:
        #face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] # x-y width height
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1), max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob) #pass blob into the gender net
        genderPrediction = genderNet.forward() #extract the prediction
        gender = genderList[genderPrediction[0].argmax()] #max prediction value (best prediction)


        ageNet.setInput(blob) #pass blob into age net
        agePrediction = ageNet.forward() #extract prediction
        age = ageList[agePrediction[0].argmax()]

        label = "{}, {}".format(gender, age)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Age-Gender", frame)
    k=cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()