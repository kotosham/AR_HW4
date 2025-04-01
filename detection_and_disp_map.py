import cv2
import numpy as np

camera_matrix = np.array([[903.23, 0, 439.84], [0, 933.83, 739.124], [0, 0, 1]])  # параметры калибровки
baseline = 0.2  # 20 см - расстояние между позициями камеры (использовалась одна камера)

# Расчет minDisparity и numDisparities
min_depth = 1.0  # минимальная ожидаемая глубина
max_depth = 3.0  # максимальная ожидаемая глубина
focal_length = camera_matrix[0, 0]

max_disparity = int((baseline * focal_length) / min_depth)
min_disparity = int((baseline * focal_length) / max_depth)
numDisparities = max_disparity - min_disparity
numDisparities = ((numDisparities + 15) // 16) * 16

# Загрузка модели YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]


def detect_people(img):
    height, width = img.shape[:2]

    # Подготовка изображения для YOLO
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Фильтрация детекций
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.5:  # class_id=0 - человек в COCO
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применяем Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return [boxes[i] for i in indices]


left_img = cv2.imread('./photo1/left_1_undistorted.jpg')
right_img = cv2.imread('./photo1/right_1_undistorted.jpg', 0)

# Создание стереоматчера
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=numDisparities,
    blockSize=8,
    P1=8 * 3 * 8**2,
    P2=32 * 3 * 8**2,
    disp12MaxDiff=5,
    uniquenessRatio=10,
    speckleWindowSize=200,
    speckleRange=3
)

# Вычисление диспарантности
disparity = stereo.compute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), right_img).astype(np.float32) / 16.0

# Детекция людей на левом изображении
people_boxes = detect_people(left_img)

# Обработка обнаруженного человека
for (x, y, w, h) in people_boxes:
    # Вычисление глубины в центре bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Усреднение диспарантности в области вокруг центра
    disparity_window = disparity[center_y - 5:center_y + 5, center_x - 5:center_x + 5]
    valid_disparities = disparity_window[disparity_window > 0]

    if len(valid_disparities) == 0:
        continue

    avg_disparity = np.mean(valid_disparities)

    # Расчет глубины
    depth = (baseline * camera_matrix[0, 0]) / avg_disparity

    # Отрисовка результатов
    cv2.rectangle(left_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_img, f"{depth:.2f}m", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


disparity_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

height, width = left_img.shape[:2]
scale = min(1920 / width, 1080 / height)
if scale < 1:
    img_resized = cv2.resize(left_img, (int(width * scale), int(height * scale)), cv2.INTER_AREA)
    disparity_resized = cv2.resize(disparity_vis,
                                   (int(width * scale),
                                    int(height * scale)),
                                   interpolation=cv2.INTER_AREA)
else:
    img_resized = left_img.copy()
    disparity_resized = disparity_vis

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result", img_resized)
cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
cv2.imshow("Disparity", disparity_resized)
cv2.resizeWindow("Disparity", disparity_resized.shape[1], disparity_resized.shape[0])
cv2.waitKey(0)
cv2.destroyAllWindows()