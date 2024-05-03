import cv2
import numpy as np
import rembg

def draw_outline(image, contours):
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

def resize_image(image, width):
    aspect_ratio = width / float(image.shape[1])
    height = int(image.shape[0] * aspect_ratio)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def main():
    input_image_path = "4.jpg"
    display_width = 800


    image = cv2.imread(input_image_path)
    if image is None:
        print("Could not read the image.")
        return

    resized_image = resize_image(image, display_width)


    cv2.imshow("Image", resized_image)


    roi_rect = cv2.selectROI("Image", resized_image)
    if sum(roi_rect) == 0:
        print("ROI selection canceled.")
        return
    x, y, w, h = map(int, roi_rect)

    roi = resized_image[y:y+h, x:x+w]

    rgba = cv2.cvtColor(roi, cv2.COLOR_BGR2RGBA)


    result = np.array(rembg.remove(rgba), dtype=np.uint8)


    result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)


    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours) > 0:
        draw_outline(roi, contours)


    cv2.imshow("Cropped Image with Outline", roi)
    cv2.waitKey(3000)

    cv2.imshow("Output", result)


    print("Final output is shown. Press q key to exit.")


    key = cv2.waitKey(0) & 0xFF


    if key == ord("q"):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
