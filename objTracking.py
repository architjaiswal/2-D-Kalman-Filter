'''
    File name         : objTracking.py
    Description       : Main file for object tracking
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Last updated      : 20/01/2023
    Python Version    : 3.9
'''

import cv2
from Detector import detect
from KalmanFilter import KalmanFilter
import matplotlib.pyplot as plt

def main():

    VideoCap = cv2.VideoCapture('video/randomball.avi')

    ControlSpeedVar = 100
    HiSpeed = 100

    KF = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)

    debugMode = 1

    # Logging arrays
    true_x, true_y = [], []
    pred_x, pred_y = [], []
    est_x, est_y = [], []

    frame_idx = []

    i = 0

    while(True):

        ret, frame = VideoCap.read()
        if not ret:
            break

        centers = detect(frame, debugMode)

        if (len(centers) > 0):

            # measured (true)
            mx, my = centers[0]

            # draw measured position
            cv2.circle(frame, (int(mx), int(my)), 10, (0,191,255), 2)

            # Predict
            (px, py) = KF.predict()

            cv2.rectangle(frame,
                          (int(px - 15), int(py - 15)),
                          (int(px + 15), int(py + 15)),
                          (255, 0, 0), 2)

            # Update
            (ux, uy) = KF.update(centers[0])

            cv2.rectangle(frame,
                          (int(ux - 15), int(uy - 15)),
                          (int(ux + 15), int(uy + 15)),
                          (0, 0, 255), 2)

            cv2.putText(frame,"Estimated Position",(int(ux+15),int(uy+10)),0,0.5,(0,0,255),2)
            cv2.putText(frame,"Predicted Position",(int(px+15),int(py)),0,0.5,(255,0,0),2)
            cv2.putText(frame,"Measured Position",(int(mx+15),int(my-15)),0,0.5,(0,191,255),2)

            # Save data
            frame_idx.append(i)

            true_x.append(float(mx))
            true_y.append(float(my))

            pred_x.append(float(px))
            pred_y.append(float(py))

            est_x.append(float(ux))
            est_y.append(float(uy))

            i += 1

        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

        cv2.waitKey(HiSpeed-ControlSpeedVar+1)

    VideoCap.release()
    cv2.destroyAllWindows()

    # -------- Plot results --------

    plt.figure()

    plt.subplot(2,1,1)
    plt.title("X Position vs Time")
    plt.plot(frame_idx, true_x, label="Measured", color='orange')
    plt.plot(frame_idx, pred_x, label="Predicted", color='blue')
    plt.plot(frame_idx, est_x, label="Estimated", color='red')
    plt.ylabel("X Position")
    plt.legend()
    plt.grid()

    plt.subplot(2,1,2)
    plt.title("Y Position vs Time")
    plt.plot(frame_idx, true_y, label="Measured", color='orange')
    plt.plot(frame_idx, pred_y, label="Predicted", color='blue')
    plt.plot(frame_idx, est_y, label="Estimated", color='red')
    plt.ylabel("Y Position")
    plt.xlabel("Frame")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()