        
import math
from modelTRT import *
class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 2  # your model classes
        self.class_names = ["CT","T"]
import argparse
import numpy as np
import mss
import cv2
import time
from pynput.mouse import Button, Controller


def main(eng_path,ajustmentY,width,height):


    pred = Predictor(engine_path=eng_path)
    mouse = Controller()
    prev_bbox = None
    prev_bbox_count = 0
    sct =mss.mss()
    monitor = {"top": 39, "left": 0, "width": width, "height": height}   
    screen_center = (monitor["width"]/2, monitor["height"]/2)
    cv2.namedWindow("Detected Boxes", cv2.WINDOW_AUTOSIZE)
    time.sleep(4)
    while True:

        start_time = time.perf_counter()

        im =sct.grab(monitor)

        image_np = cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2RGB)

        data = pred.infer(image_np)
        end2end=True 
        ratio = min(640 / image_np.shape[0], 640 / image_np.shape[1])
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)

        #boxes
        final_boxes, final_scores, final_cls_inds = dets[:,:4], dets[:, 4], dets[:, 5]
        print(final_boxes)

            # If there are no bounding boxes, skip the rest of the loop
        if len(final_boxes) > 0:
            
        
          # Initialize a variable to store the closest bounding box to the center of the screen
            closest_bbox = None
            closest_distance = 900000
            
            # Iterate through the bounding boxes
            for bbox in final_boxes:
                # Get the center of the bounding box
                bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                
                # Calculate the distance between the center of the bounding box and the center of the screen
                distance = math.sqrt((bbox_center[0] - screen_center[0]) ** 2 + (bbox_center[1] - screen_center[1]) ** 2)
                
                # If this bounding box is closer to the center of the screen than the previous closest bounding box
                if distance < closest_distance:
                    # Update the closest bounding box
                    closest_bbox = bbox
                    closest_distance = distance
                    
                    # If this bounding box is the same as the previously selected bounding box
                elif np.allclose(bbox, prev_bbox):
                    # Increase the count for the number of times this bounding box has been selected
                    prev_bbox_count += 1
                    # Update the closest bounding box to be the previously selected bounding box
                    # using an exponential function to prioritize the previously selected bounding box
                    closest_bbox = bbox
                    closest_distance = distance * 0.9 ** prev_bbox_count
            
            # Get the center of the closest bounding box
            xmin, ymin, xmax, ymax = closest_bbox
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            print(x,y)

        
            mouse.position = (x,y+ajustmentY)

            mouse.press(Button.left)
            time.sleep(0.01) # 10ms
            mouse.release(Button.left)
            time.sleep(0.01) # 10ms
            
            # Update the previously selected bounding box
            prev_bbox = closest_bbox



            image_np = vis(image_np, final_boxes, final_scores, final_cls_inds,conf=0.2, class_names=["CT","T"])
            #print(origin_img.shape)
        cv2.imshow("Detected Boxes",image_np)
        #cv2.resizeWindow("image", 1280,720)
        print("Time taken: ", (time.perf_counter() - start_time)*1000, "ms")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            
            
            break

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Atlas csgo bot.')
    parser.add_argument("-e", required=True, type=str, help="path to engine")
    parser.add_argument("-a", default=10, type=int, help="adjustment to the Y coordinate of mouse position")
    parser.add_argument("-x", default=1280, type=int, help="Width in pixels of game window")
    parser.add_argument("-y", default=720, type=int, help="Height in pixels of game window")
    
    
    args = parser.parse_args()
 
    main( args.e,args.a,args.x,args.y)
