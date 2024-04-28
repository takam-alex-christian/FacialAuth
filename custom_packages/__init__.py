
import os

import cv2

import mtcnn
import numpy as np

from PIL import Image

mtcnn_detector = mtcnn.MTCNN()

class ImageProcessor(object):
    
    face_detector = None
    
    image_src = ""
    output_folder = ""
    
    image_raw = None
    
    found_face_data = None
    
    processed_image = None
    
    
    
    
    def __init__(self, src,output_folder, detector):
        self.src = src
        self.output_folder = output_folder
        self.face_detector = detector
        
        #load image
        self.image_raw = cv2.imread(src)
        
        
    def get_face(self):
        
        
        #first face found
        found_faces_data = self.face_detector(self.image_raw)
        
        is_face_found = len(found_faces_data) > 0
        
        if is_face_found:
            self.found_face_data = found_faces_data[0]

        return found_faces_data[0] if is_face_found else None
    

    def crop_to_facebox(self):
        x,y,w,h = self.found_face_data
        
        #cropping the image at face box coordinates
        cropped_frame = self.image_raw[y:y+h, x:x+w]
        
        self.processed_image = cropped_frame
        
        return cropped_frame

        

    #face alignment function
    def align_face(self):
        left_eye = self.found_face_data["keypoints"]["left_eye"]
        right_eye = self.found_face_data["keypoints"]["right_eye"]
        
        #aligning the face eyes
        a = left_eye[1] - right_eye[1] # can be a positive value or a negative value = y1-y2
        b = right_eye[0] - left_eye[0] # always positive => x2-x1
        
        
        #determining the angle between the two eyes vectors
        angle = np.arctan(a/b) # in radian
        rotation_angle = np.rad2deg(angle)
        
        #rotating the image by the angle 
        pil_image = Image.fromarray(self.processed_image)
        aligned_image = np.array(pil_image.rotate(-1*rotation_angle))

        return aligned_image
    
    #image resize function
    def image_resize(self,target_size=160):
        #
        (image_height, image_width) = self.processed_image.shape[:2]

        k = image_width/float(image_height) #proportionality constant
                    
        if image_height > image_width:
            resized_image = cv2.resize(self.processed_image, (int(k*target_size),target_size),interpolation=cv2.INTER_AREA)
        elif image_height < image_width:
            resized_image = cv2.resize(self.processed_image, (target_size, int(target_size/k)), interpolation=cv2.INTER_AREA)
        else:
            resized_image = self.processed_image
        
        self.processed_image = resized_image
        
        return resized_image
    
    def store_processed_image(self):
        cv2.imwrite(os.path.join(self.output_folder, os.path.basename(self.src)))