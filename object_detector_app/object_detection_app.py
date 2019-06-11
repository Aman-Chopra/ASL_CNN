import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
import imutils

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'test_model_1.pb')

NUM_CLASSES = 90
IMAGE_DIMS = (32, 32, 3)


def detect_objects(image_np, detection_graph, sess):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
   
   #cv2.rectangle(image_np,(10,10),(410,410),(0,255,0),3)
   #frame1 = image_np[10:410, 10:410]
    top, right, bottom, left = 10, 450, 325, 690
    cv2.rectangle(image_np, (left, top), (right, bottom), (0,255,0), 2)
    frame1 = image_np[top:bottom, right:left]
    frame1 = cv2.resize(frame1, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    frame1 = frame1.astype("float") / 255.0
    #frame1 = cv2.flip(frame1, 1)
    image_np_expanded = np.expand_dims(frame1, axis=0)
    

    conv2d_1_input = detection_graph.get_tensor_by_name('conv2d_1_input:0')
    output_tensor = detection_graph.get_tensor_by_name('activation_7/Softmax:0')
    
    output_tensor = sess.run(output_tensor,feed_dict={conv2d_1_input: image_np_expanded})
    print(output_tensor)
    index_min = np.argmax(output_tensor)
    print(index_min)
    print(output_tensor[0][index_min])
    
    if index_min == 0:
        alphabet  = "A"
    elif index_min == 1:
        alphabet  = "B"
    elif(index_min == 2):
        alphabet  = "C"
    elif(index_min == 3):
        alphabet  = "D"
    elif(index_min == 4):
        alphabet  = "E"
    elif(index_min == 5):
        alphabet  = "F"
    elif(index_min == 6):
        alphabet  = "G"
    elif(index_min == 7):
        alphabet  = "H"
    elif(index_min == 8):
        alphabet  = "I"
    elif(index_min == 9):
        alphabet  = "J"
    elif(index_min == 10):
        alphabet  = "K"
    elif(index_min == 11):
        alphabet  = "L"
    elif(index_min == 12):
        alphabet  = "M"
    elif(index_min == 13):
        alphabet  = "N"
    elif(index_min == 14):
        alphabet  = "O"
    elif(index_min == 15):
        alphabet  = "P"
    elif(index_min == 16):
        alphabet  = "Q"
    elif(index_min == 17):
        alphabet  = "R"
    elif(index_min == 18):
        alphabet  = "S"
    elif(index_min == 19):
        alphabet  = "T"
    elif(index_min == 20):
        alphabet  = "U"
    elif(index_min == 21):
        alphabet  = "V"
    elif(index_min == 22):
        alphabet  = "W"
    elif(index_min == 23):
        alphabet  = "X"
    elif(index_min == 24):
        alphabet  = "Y"
    elif(index_min == 25):
        alphabet  = "Z"
    elif(index_min == 26):
        alphabet  = "Delete"
    elif(index_min == 27):
        alphabet  = "Nothing"
    elif(index_min == 28):
        alphabet  = "Space"

    print(alphabet)
    cv2.putText(img = image_np, text = alphabet, org = (400,400), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 3,
            color = (0, 255, 0))

    #print("PHONCHA CHALO YAHAN TAK TO")
    return image_np


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame, detection_graph,sess))

    fps.stop()
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()
    
    while True:  # fps._numFrames < 120
        #frame1 = cv2.imread("/Users/siddharthasingh/Desktop/del_test_1.jpg")
        frame1 = video_capture.read()
        frame1 = imutils.resize(frame1, width=700)
        frame1 = cv2.flip(frame1, 1)
        (height, width) = frame1.shape[:2]
        """cv2.rectangle(frame1,(10,10),(310,310),(0,255,0),3)
        frame2 = frame1[10:310, 10:310]
        frame2 = cv2.resize(frame2, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        frame2 = frame2.astype("float") / 255.0"""
        input_q.put(frame1)

        t = time.time()

        #output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', output_q.get())
        
        fps.update()
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
