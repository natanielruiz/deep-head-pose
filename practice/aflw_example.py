#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
# website: http://mpatacchiola.github.io/
# email: massimiliano.patacchiola@plymouth.ac.uk
# Python code for information retrieval from the Annotated Facial Landmarks in the Wild (AFLW) dataset.
# In this example the faces are isolated and saved in a specified output folder.
# Some information (roll, pitch, yaw) are returned, they can be used to filter the images.
# This code requires OpenCV and Numpy. You can easily bypass the OpenCV calls if you want to use
# a different library. In order to use the code you have to unzip the images and store them in
# the directory "flickr" mantaining the original folders name (0, 2, 3).
#
# The following are the database properties available (last updated version 2012-11-28):
#
# databases: db_id, path, description
# faceellipse: face_id, x, y, ra, rb, theta, annot_type_id, upsidedown
# faceimages: image_id, db_id, file_id, filepath, bw, widht, height
# facemetadata: face_id, sex, occluded, glasses, bw, annot_type_id
# facepose: face_id, roll, pitch, yaw, annot_type_id
# facerect: face_id, x, y, w, h, annot_type_id
# faces: face_id, file_id, db_id
# featurecoords: face_id, feature_id, x, y
# featurecoordtype: feature_id, descr, code, x, y, z

import sqlite3
import cv2
import os.path
import numpy as np

#Change this paths according to your directories
images_path = "./flickr/"
storing_path = "./output/"

def main():

    #Image counter
    counter = 1

    #Open the sqlite database
    conn = sqlite3.connect('aflw.sqlite')
    c = conn.cursor()

    #Creating the query string for retriving: roll, pitch, yaw and faces position
    #Change it according to what you want to retrieve
    select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h"
    from_string = "faceimages, faces, facepose, facerect"
    where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id"
    query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

    #It iterates through the rows returned from the query
    for row in c.execute(query_string):

        #Using our specific query_string, the "row" variable will contain:
        # row[0] = image path
        # row[1] = face id
        # row[2] = roll
        # row[3] = pitch
        # row[4] = yaw
        # row[5] = face coord x
        # row[6] = face coord y
        # row[7] = face width
        # row[8] = face heigh

        #Creating the full path names for input and output
        input_path = images_path + str(row[0])
        output_path = storing_path + str(row[0])

        #If the file exist then open it       
        if(os.path.isfile(input_path)  == True):
            #image = cv2.imread(input_path, 0) #load in grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #load the colour version

            #Image dimensions
            image_h, image_w = image.shape
            #Roll, pitch and yaw
            roll   = row[2]
            pitch  = row[3]
            yaw    = row[4]
            #Face rectangle coords
            face_x = row[5]
            face_y = row[6]
            face_w = row[7]
            face_h = row[8]

            #Error correction
            if(face_x < 0): face_x = 0
            if(face_y < 0): face_y = 0
            if(face_w > image_w): 
                face_w = image_w
                face_h = image_w
            if(face_h > image_h): 
                face_h = image_h
                face_w = image_h

            #Crop the face from the image
            image_cropped = np.copy(image[face_y:face_y+face_h, face_x:face_x+face_w])
            #Uncomment the lines below if you want to rescale the image to a particular size
            #to_size = 64
            #image_rescaled = cv2.resize(image_cropped, (to_size,to_size), interpolation = cv2.INTER_AREA)
            #Uncomment the line below if you want to use adaptive histogram normalisation
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
            #image_normalised = clahe.apply(image_rescaled)
            #Save the image
            #change "image_cropped" with the last uncommented variable name above
            cv2.imwrite(output_path, image_cropped)

            #Printing the information
            print "Counter: " + str(counter)
            print "iPath:    " + input_path
            print "oPath:    " + output_path
            print "Roll:    " + str(roll)
            print "Pitch:   " + str(pitch)
            print "Yaw:     " + str(yaw)
            print "x:       " + str(face_x)
            print "y:       " + str(face_y)
            print "w:       " + str(face_w)
            print "h:       " + str(face_h)
            print ""

            #Increasing the counter
            counter = counter + 1 

        #if the file does not exits it return an exception
        else:
            raise ValueError('Error: I cannot find the file specified: ' + str(input_path))

    #Once finished the iteration it closes the database
    c.close()

if __name__ == "__main__":
    main()

