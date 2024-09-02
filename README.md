This is a system design for the visually impaired to recognize the sidewalk based on yolov5(temporarily)
It has solve some problems:
1、It can recognize the traffic light(red and green)
2、It can recognize the roadline(although it is toxic)
3、When it recognize the red signal traffic light(or the green one),it can tell the user the light it recognize

The problem we face:
1、the sidewalk recognition is too terrible that it can't use in real world
2、the sound from the function like a robot
3、the data we train in yolo is too small for real traffic situation
4、THE SYSTEM HAS TOXIC PERFORMANCE IN RUNNING
5、THE LOGIC IN RUNNING HAS SOME PROBLEM. WHEN IT CANT RECOGNIZE SOMETHING THEN IT WILL NOT UPDATE THE STREAM, IT LET THE VIDEO PHASE
6、WE WANT IT RUN IN MOBLIE DEVICE IN A TOLERABLE SPEED
7、IT DOESNT HAVE THE JUDGE SYSTEM TO MAKE DECISIONS FOR THE Visually impaired
I think the most significant thing is 1, 2, 3 
AT LEAST,WE NEED IT CAN RUN IN PROPER WAY
