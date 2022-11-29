# -*- coding: utf-8 -*-
"""
@file    : ARBA-LIVE
@brief   : Live streaming version of ARBA neural.
@date    : 2022/11/29
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bug     : None.
"""


import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#st.set_page_config(layout="wide")
st.title("ABMA Live")

def video_frame_callback(frame):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = frame.to_ndarray(format="bgr24")
        results = pose.process(image)
        image.flags.writeable = True
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        flipped = np.flip(image, axis=1)#img[::-1,:,:] if flip else img
        return av.VideoFrame.from_ndarray(flipped, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)