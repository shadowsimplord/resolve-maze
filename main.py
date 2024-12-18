import cv2
import numpy as np
from queue import PriorityQueue
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Tạo hộp thoại chọn file
Tk().withdraw()
file_path = askopenfilename()

# Kiểm tra xem người dùng đã chọn file chưa
if not file_path:
    print("Không có tệp nào được chọn.")
    exit()

# Tải hình ảnh mê cung
image = cv2.imread(file_path)

# Kiểm tra xem hình ảnh có được tải thành công không
if image is None:
    print("Không thể mở hoặc đọc tệp hình ảnh. Kiểm tra đường dẫn và tên tệp.")
    exit()

# Chuyển đổi hình ảnh sang không gian màu HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Xác định điểm bắt đầu (màu đỏ)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
start_mask1 = cv2.inRange(hsv, lower_red, upper_red)
lower_red = np.array([160, 100, 100])
upper_red = np.array([179, 255, 255])
start_mask2 = cv2.inRange(hsv, lower_red, upper_red)
start_mask = start_mask1 + start_mask2
start_points = cv2.findNonZero(start_mask)

if start_points is not None:
    start_point = (start_points[0][0][1], start_points[0][0][0])  # (y, x)
else:
    print("Không tìm thấy điểm bắt đầu màu đỏ")
    exit()

# Xác định điểm kết thúc (màu xanh lá cây)
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])
end_mask = cv2.inRange(hsv, lower_green, upper_green)
end_points = cv2.findNonZero(end_mask)

if end_points is not None:
    end_point = (end_points[0][0][1], end_points[0][0][0])  # (y, x)
else:
    print("Không tìm thấy điểm kết thúc màu xanh lá cây")
    exit()

# Xác định tường (màu đen)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
grid = binary // 255  # 0: đường đi, 1: tường

# Triển khai thuật toán A*
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, end):
    rows, cols = grid.shape
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    cost_so_far = {start: 0}
    while not open_set.empty():
        _, current = open_set.get()
        if current == end:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        neighbors = [(current[0]+dy, current[1]+dx) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]]
        for neighbor in neighbors:
            ny, nx = neighbor
            if 0 <= ny < rows and 0 <= nx < cols and grid[ny][nx] == 0:
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(end, neighbor)
                    open_set.put((priority, neighbor))
                    came_from[neighbor] = current
    return None

# Tìm đường đi ngắn nhất
path = astar(grid, start_point, end_point)

# Vẽ đường đi lên hình ảnh mê cung
if path:
    # Vẽ đường đi bằng cách nối các điểm
    thickness = 15  # Đặt độ dày của đường vẽ tại đây
    for i in range(len(path)-1):
        point1 = (path[i][1], path[i][0])      # Chuyển từ (y,x) sang (x,y) cho cv2.line
        point2 = (path[i+1][1], path[i+1][0])
        cv2.line(image, point1, point2, (0, 255, 0), thickness=thickness)  
    
    # Chuyển đổi hình ảnh từ BGR sang RGB để hiển thị bằng matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Hiển thị hình ảnh gốc và hình ảnh đã giải
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB))
    plt.title('Hình ảnh gốc')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_rgb)
    plt.title('Hình ảnh đã giải')
    
    plt.show()
else:
    print("Không tìm thấy đường đi")