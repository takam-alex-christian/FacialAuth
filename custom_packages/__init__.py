
import os

import cv2

import mtcnn

import numpy as np

from PIL import Image


class ImageProcessor(object):
    
    face_detector = mtcnn.MTCNN()
    
    image_src = ""
    
    output_folder = ""
    
    image_raw = None
    
    found_face_data = None
    
    processed_image = np.array([])
    
    
    def __init__(self, src="",output_folder="", keep_ratio= False):
        self.src = src
        self.output_folder = output_folder
        
        if len(self.output_folder) != 0:
            if not(os.path.exists(self.output_folder)):
                os.mkdir(self.output_folder)
        
        #load image
        self.image_raw = cv2.imread(src)
        self.processed_image = np.array(self.image_raw)
        
        
        #detect face
        self.get_face()
        
        if self.found_face_data is not None:
            #crop face     
            self.crop_to_facebox()
            
            #align face
            self.align_face()
            
            #resize
            self.image_resize(target_size=160, keep_ratio=keep_ratio)
            
            if len(self.output_folder) != 0:
                #store image
                self.store_processed_image()
    
    def set_imraw(self, im_data):
        self.image_raw = im_data
        self.processed_image = np.array(self.image_raw)
    

    def get_face(self):
        
        
        #first face found
        found_faces_data = self.face_detector.detect_faces(self.image_raw)
        
        is_face_found = len(found_faces_data) > 0
        
        if is_face_found:
            self.found_face_data = found_faces_data[0]
            
            print(self.found_face_data)
            print(self.output_folder)

        return found_faces_data[0] if is_face_found else None
    

    def crop_to_facebox(self):
        
        if not (self.found_face_data):
            return self.processed_image
        
        x,y,w,h = self.found_face_data["box"]
        
        #cropping the image at face box coordinates
        cropped_frame = self.image_raw[y:y+h, x:x+w]
        
        self.processed_image = cropped_frame
        
        return cropped_frame

        

    #face alignment function
    def align_face(self):
        
        if not(self.found_face_data):
            return self.processed_image
        
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
        
        self.processed_image = aligned_image

        return aligned_image
    
    #image resize function
    def image_resize(self,target_size=160, keep_ratio = True):
        #
        (image_height, image_width) = self.processed_image.shape[:2]

        k = image_width/float(image_height) #proportionality constant
        
        if keep_ratio:
                    
            if image_height > image_width:
                resized_image = cv2.resize(self.processed_image, (int(k*target_size),target_size),interpolation=cv2.INTER_AREA)
            elif image_height < image_width:
                resized_image = cv2.resize(self.processed_image, (target_size, int(target_size/k)), interpolation=cv2.INTER_AREA)
            else:
                resized_image = self.processed_image
            
            
        else:
            resized_image = cv2.resize(self.processed_image, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        self.processed_image = resized_image
        
        return resized_image
    
    def get_reshaped_dims(self):
        self.processed_image = np.expand_dims(self.processed_image,axis=0)
        return self.processed_image
    
    def store_processed_image(self):
        cv2.imwrite(os.path.join(self.output_folder, os.path.basename(self.src)), self.processed_image)
        return self.processed_image
    
    def store_processed_image_to(self, output_folder, file_id):
        cv2.imwrite(f"{output_folder}\\{file_id}.jpg", self.processed_image)
        return self.processed_image
    
    def get_processed_image(self): # just returns processed image as a numpy array
        return self.processed_image
    
    def show_processed_image(self):
        
        if np.size(self.processed_image) > 0:
            cv2.imshow("processed image", self.processed_image)
        
            
        else:
            cv2.imshow("only raw available", self.image_raw)
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def cos_similarity(source_emb, test_emb):
    """

    """
    
    a = np.matmul(np.transpose(source_emb), test_emb)
 
    b = np.matmul(np.transpose(source_emb), source_emb)
    c = np.matmul(np.transpose(test_emb), test_emb)
    
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
        
        