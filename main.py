import cv2
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
import os
from util import get_parking_boxes,empty_or_not
import streamlit as st

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


def get_masked_parking_components(mask_url):
    #mask='mask_1920_1080.png'
    
    mask=cv2.imread(mask_url,0)
    #the connected component uses the concept of piecewise connected graphs
    #connected component labeling
    #depth-first search (DFS) or breadth-first search (BFS) 
    conn_components=cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)
    return get_parking_boxes(conn_components)


def returnSpotStatus(x1,y1,w,h,frame):

     #draws a rectangle according to the coordinates we returned
    #(255,0,0)->Blue
    croped_image=frame[y1:y1+h,x1:x1+w,:]

    return empty_or_not(croped_image)



def draw_image(spot,frame,status):
    x1,y1,w,h=spot
    if status:

        cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)
    else:
        cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,255),2)



def updateDifFromPrevFrame(prev_frame,spots,frame,diffs):
            #Checking the difference in parking spot from the previous frame
   
    for spot_idx,_ in enumerate(spots):
        x1,y1,w,h=spots[spot_idx]
        croped_image=frame[y1:y1+h,x1:x1+w,:]  
        diffs[spot_idx]=calc_diff(croped_image,prev_frame[y1:y1+h,x1:x1+w,:] )
    return diffs



def updateSpotStatus(spots,diffs,spot_status,frame,prev_frame):
    if prev_frame is None:
        arr_=range(len(spots))
    else:
        #np.percentile(diffs, 95)
        arr_=[j for j in np.argsort(diffs) if (diffs[j]/np.amax(diffs))>.4]
    for spot_idx in arr_:
        x1,y1,w,h=spots[spot_idx]
        spot_status[spot_idx]=returnSpotStatus(x1,y1,w,h,frame)
    prev_frame=frame.copy()
    return spot_status


def getRealTimeUpdate(video_url,mask_url):

    cap=cv2.VideoCapture(video_url)
    spots=get_masked_parking_components(mask_url)
   
    ret=True

    #Update the status in every 30 frames
    step=50
    frame_num=0
    spot_status=[None for j in spots]

    #Compare frames to identify  the changing spots
    diffs=[None for j in spots]
    prev_frame= None
    # Create a video file to store the frames
    placeholder=st.empty()

    while ret:
        ret,frame=cap.read()

        if frame_num % step==0 and prev_frame is not None:
            diffs=updateDifFromPrevFrame(prev_frame,spots,frame,diffs)

        #Checking and updating once in 30 frames
        if frame_num % step==0:
            spot_status=updateSpotStatus(spots,diffs,spot_status,frame,prev_frame)
                
        #Visualize
        for spot_idx,_ in enumerate(spots):
            
            draw_image(spots[spot_idx],frame,spot_status[spot_idx])


        #Fit to screen for better visualization
        #cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        #cv2.imshow(f'frame',frame)
 
        placeholder.image(frame, use_column_width=True, channels="BGR")

        if cv2.waitKey(25) & 0xFF==ord('q'):
            break
        frame_num+=1
    cap.release()


def plotThedifference(diffs):
    # Sort the differences in descending order
    percentile_85 = np.percentile(diffs, 85)

    # Filter points with differences greater than the 85th percentile
    points_above_85th_percentile = [(i, diff) for i, diff in enumerate(diffs) if diff > percentile_85]

    # Print or use the points as needed
    print("Points with differences greater than 85th percentile:")
    print(points_above_85th_percentile)

    # Create a red boxplot highlighting points above the 85th percentile
    plt.boxplot(diffs, flierprops=dict(markerfacecolor='red', marker='o'))
    plt.title('Boxplot of Differences')
    plt.ylabel('Differences')
    plt.show()




