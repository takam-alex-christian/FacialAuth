import os

import custom_packages

import mtcnn

#create a detector
face_detector = mtcnn.MTCNN()


output_folder = f"{os.path.join(os.getcwd(), "test","test_out")}"
src = f"{os.path.join(os.getcwd(), "test","test_in", "im1.jpg")}"

im_processor = custom_packages.ImageProcessor(output_folder= output_folder, src=src, detector=face_detector)
