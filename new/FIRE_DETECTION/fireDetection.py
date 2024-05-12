import cv2           

fire_cascade = cv2.CascadeClassifier('fire_detection_cascade_model.xml') 

cap = cv2.VideoCapture(0) 
detect = False 
# dem_lua = 0
		
while(True):
    Alarm_Status = False
    ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5) 

    for (x,y,w,h) in fire:
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        if detect == False:
            print("Khong co lua")
            detect = True
            global num_lua 
            numlua = "0"
        if detect == True:
            print("Co lua")
            detect = True
            num_lua = "2"
            print(num_lua)
    num_lua = "0"
    print(num_lua)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
