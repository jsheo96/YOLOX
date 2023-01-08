import cv2
import argparse
import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", default="C:/Records/Local Records/Ch1_169.254.152.184/20221227151220082.avi",
        help="path to images or video"
    )
    parser.add_argument(
        "--path2", default="C:/Records/Local Records/Ch1_169.254.152.185/20221227151222090.avi",
        help="path to images or video 2"
    )
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.path)
    cap2 = cv2.VideoCapture(args.path2)
    lag = 47
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap2.get(cv2.CAP_PROP_POS_FRAMES) + lag)
    while True:
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        if not ret or not ret2:
            break
        result = np.hstack((frame, frame2))
        cv2.imshow('', cv2.resize(result, None, fx=0.25, fy=0.25))
        cv2.waitKey()

