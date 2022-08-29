"""
Example usage of the Palette Library
"""
import Palette
import cv2

cap = cv2.VideoCapture(0)
photoreceptor = Palette.LMC(100, 100)

while cap.isOpened():
    ret, frames = cap.read()
    fig = Palette.optic_blur(frames, 15)
    photoreceptor.update(fig)
    cv2.imshow('neuron map', cv2.resize(photoreceptor.lmc_output, (400, 400)))
    cv2.imshow('filtered', cv2.resize(photoreceptor.filtered_data, (400, 400)))
    cv2.imshow('frame', fig)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
