import cv2
import numpy as np
import glob

# Путь к изображению, которое нужно откалибровать
distorted_img_filename = "./photo1/right_3.jpg"

# Размеры шахматной доски
number_of_squares_X = 10  # Количество квадратов шахматной доски по оси X
number_of_squares_Y = 7  # Количество квадратов шахматной доски по оси Y
nX = number_of_squares_X - 1  # Количество внутренних углов по оси X
nY = number_of_squares_Y - 1  # Количество внутренних углов по оси Y
square_size = 0.016  # Длина стороны квадрата в метрах

# Сохраняем векторы 3D точек для всех изображений шахматной доски (в мировой системе координат)
object_points = []

# Сохраняем векторы 2D точек для всех изображений шахматной доски (в системе координат камеры)
image_points = []

# Устанавливаем критерии завершения. Остановка произойдёт либо при достижении точности, либо при завершении определённого количества итераций.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Определяем реальные мировые координаты для точек в 3D системе координат
# Объектные точки: (0,0,0), (1,0,0), (2,0,0) ...., (5,8,0)
object_points_3D = np.zeros((nX * nY, 3), np.float32)

# Это координаты x и y
object_points_3D[:, :2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2)

object_points_3D = object_points_3D * square_size


def main():
    # Получаем пути к изображениям в текущей директории
    images = glob.glob('./photo1/*.jpg')

    # Проходимся по каждому изображению шахматной доски
    for image_file in images:

        # Загружаем изображение
        image = cv2.imread(image_file)

        # Конвертируем изображение в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Находим углы шахматной доски
        success, corners = cv2.findChessboardCorners(gray, (nY, nX), None)

        # Если углы найдены алгоритмом, рисуем их
        if success == True:
            # Добавляем объектные точки
            object_points.append(object_points_3D)

            # Находим более точные пиксели углов
            corners_2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Добавляем точки изображения
            image_points.append(corners_2)

            # Рисуем углы
            cv2.drawChessboardCorners(image, (nY, nX), corners_2, success)


    # Теперь берём искажённое изображение и исправляем искажения
    distorted_image = cv2.imread(distorted_img_filename)

    # Выполняем калибровку камеры, возвращая матрицу камеры, коэффициенты искажений, векторы поворота и переноса и т. д.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points,
                                                       image_points,
                                                       gray.shape[::-1],
                                                       None,
                                                       None)

    # Получаем размеры изображения
    height, width = distorted_image.shape[:2]

    # Оптимизируем матрицу камеры
    # Возвращает оптимальную матрицу камеры и прямоугольную область интереса
    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist,
                                                               (width, height),
                                                               1,
                                                               (width, height))

    # Исправляем искажения изображения
    undistorted_image = cv2.undistort(distorted_image, mtx, dist, None,
                                      optimal_camera_matrix)



    # Отображаем ключевые параметры, полученные в процессе калибровки камеры
    print("Оптимальная матрица камеры:")
    print(optimal_camera_matrix)

    print("\n Коэффициент искажения:")
    print(dist)

    print("\n Векторы поворота:")
    print(rvecs)

    print("\n Векторы переноса:")
    print(tvecs)

    # Создаём имя для выходного файла, убрав '.jpg' из исходного имени
    size = len(distorted_img_filename)
    new_filename = distorted_img_filename[:size - 4]
    new_filename = new_filename + '_undistorted.jpg'

    # Сохраняем исправленное изображение
    cv2.imwrite(new_filename, undistorted_image)

    cv2.destroyAllWindows()

main()