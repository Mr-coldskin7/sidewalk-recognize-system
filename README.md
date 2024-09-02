# Sidewalk Recognition System for the Visually Impaired

## Overview
This project aims to develop a system that assists visually impaired individuals in recognizing sidewalks and navigating urban environments. The system utilizes YOLOv5 for object detection, focusing on traffic lights and road lines, while also attempting to identify sidewalks.

## Features
1. **Traffic Light Recognition**: 
   - Successfully identifies red and green traffic lights.
   - Provides audio feedback to the user regarding the recognized signal.

2. **Road Line Detection**: 
   - Capable of recognizing road lines, although performance is currently suboptimal.

## Current Challenges
Despite the initial successes, several significant issues need to be addressed:

1. **Sidewalk Recognition**: 
   - The current performance of sidewalk recognition is inadequate for real-world application.

2. **Audio Feedback Quality**: 
   - The audio output sounds robotic, which may hinder user experience.

3. **Insufficient Training Data**: 
   - The dataset used for training the YOLO model is too small to effectively handle real traffic situations.

4. **Toxic Performance**: 
   - The system exhibits erratic behavior during operation, leading to unreliable performance.

5. **Logic and Stream Update Issues**: 
   - The system fails to update the video stream when it cannot recognize an object, causing the video feed to freeze.

6. **Mobile Device Compatibility**: 
   - The system needs to run efficiently on mobile devices while maintaining acceptable speed.

7. **Decision-Making Capability**: 
   - The system lacks a robust decision-making framework to assist visually impaired users in navigating their environment.

## Next Steps
To improve the system, we need to focus on the following areas:

- Enhance sidewalk recognition capabilities to ensure reliable real-world application.
- Improve the quality of audio feedback to provide a more natural user experience.
- Expand the training dataset to include a wider variety of traffic scenarios and environments.
- Address performance issues to ensure smooth operation without toxic behavior.
- Implement a logic system that allows for continuous video stream updates, even when objects are not recognized.
- Optimize the system for mobile devices to ensure it runs at a tolerable speed.
- Develop a decision-making framework to assist users in navigating their surroundings effectively.


