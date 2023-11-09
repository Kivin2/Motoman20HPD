from robolink import *    # RoboDK's API
from robodk import *      # Math toolbox for robots
from ultralytics import YOLO
import cv2
import math
RDK = Robolink()
robot = RDK.Item('', ITEM_TYPE_ROBOT)
target = RDK.Item('Target 1')
target2 = RDK.Item('Target 2')
target3 = RDK.Item('Target 3')
target4 = RDK.Item('Target 5')
target5 = RDK.Item('Target 6')
target6 = RDK.Item('Target 7')
target7 = RDK.Item('Target 8')
target8 = RDK.Item('Target 9')
target9 = RDK.Item('Target 10')
home = RDK.Item('Home')
program = RDK.Item("Prog1", ITEM_TYPE_PROGRAM)
target_pose = target.Pose()
xyz_ref = target_pose.Pos()
robot.MoveJ(home)
cap = cv2.VideoCapture(0)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

model = YOLO("C:/Users/Kevin/Desktop/ultralytics/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

program_ejecutado = False

while True:
    success, img = cap.read()
    # Doing detections using YOLOv8 frame by frame
    # stream=True will use the generator and it is more efficient than normal
    results = model(img, stream=True)
    
    # Reiniciar el programa si se detecta otro objeto que no sea persona
    if not any(class_name == "person" for class_name in classNames):
        program_ejecutado = False
    
    # Once we have the results we can check for individual bounding boxes and see how well it performs
    # Once we have have the results we will loop through them and we will have the bounding boxes for each of the result
    # we will loop through each of the bounding box
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # print(x1, y1, x2, y2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # print(box.conf[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            # print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            
            # Verificar si se detecta una botella y ejecutar el programa solo una vez
            if class_name == "person":
                robot.MoveJ(target)
                robot.MoveJ(target2)
                robot.MoveJ(target3)
                robot.RunInstruction('Program_Done')

            if class_name == "cell phone":
                robot.MoveJ(target4)
                robot.MoveJ(target5)
                robot.MoveJ(target6)
                robot.RunInstruction('Program_Done')                

            if class_name == "bottle":
                robot.MoveJ(target7)
                robot.MoveJ(target8)
                robot.MoveJ(target9)
                robot.RunInstruction('Program_Done')                  

            #if class_name == "person" and not program_ejecutado:
             #   program.RunProgram()
              #  robot.RunInstruction('Program_Done')
               # program_ejecutado = True  # Establecer la variable de estado como True

    # out.write(img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()

robot.MoveJ(home)
