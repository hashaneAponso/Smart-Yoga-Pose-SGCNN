import cv2

def draw_bbox_landmarks(image, bbox=[], landmarks=[]):
   
    height, width, _ = image.shape
    # Convert the keypoints coordinates to pixel coordinates
    keypoints = [(int(x * width), int(y * height)) for x, y in list(landmarks.values())]

    # Draw an "x" at each keypoint coordinate on the image
    color = (0, 255, 255)  # Yellow
    thickness = 1
    for x, y in keypoints:
        cv2.line(image, (x-2, y-2), (x+2, y+2), color, thickness)
        cv2.line(image, (x+2, y-2), (x-2, y+2), color, thickness)


    # Draw Bounding Box
    # Find extreme x and y coordinates of landmarks
    x_coords = [keypoint[0] for keypoint in keypoints]
    y_coords = [keypoint[1] for keypoint in keypoints]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    scale_factor = 1.5
    x_min = int(x_min - (x_max - x_min) * (scale_factor - 1) / 2)
    x_max = int(x_max + (x_max - x_min) * (scale_factor - 1) / 2)
    y_min = int(y_min - (y_max - y_min) * (scale_factor - 1) / 2)
    y_max = int(y_max + (y_max - y_min) * (scale_factor - 1) / 2)

    # Limit bounding box to image boundaries
    x_min = max(10, x_min)
    x_max = min(width-10, x_max)
    y_min = max(10, y_min)
    y_max = min(height, y_max-10)

    # Draw bounding box around landmarks
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # # Show the image
    # cv2.imshow('Bounding Box and Keypoints', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

if __name__ == "__main__":

    from pipeline import pipeline

    img = cv2.imread("D:/Hashane/MSC/Module/research 2022/work/code/New folder/1/Client-Yoga-Pose-SGCNN/Dataset/Extracted_Images/Abhay_Bhujangasana/Abhay_Bhujangasana_image_5.2s.jpg")
    # img = cv2.imread("C:/Users/Lasal Jayawardena/Documents/Projects/Spatial_CNN_Yoga_Project/Dataset/Extracted_Images/Ameya_Tadasana/Ameya_Tadasana_image_5.1s.jpg")

    try:
        result = pipeline(img)
        print(result[2])
    except Exception as e:
        print(e)

    try:
        image = draw_bbox_landmarks(img, result[1], result[2])
        print(image)
    except Exception as e:
        print(e)
