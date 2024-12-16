import numpy as np
import imageio.v2 as imageio
from skimage.transform import resize
import matplotlib.pyplot as plt

img = imageio.imread('maze22.png')

# Kiểm tra và chuyển đổi hình ảnh sang RGB nếu cần
if img.shape[2] == 4:  # Nếu hình ảnh có 4 kênh (RGBA)
    img = img[:, :, :3]  # Chỉ giữ lại 3 kênh đầu (RGB)

img_tinted = img * [1, 0.95, 0.9]

# Hiển thị hình ảnh gốc
plt.subplot(1, 2, 1)
plt.imshow(img)

# Hiển thị hình ảnh đã được tô màu
plt.subplot(1, 2, 2)
plt.imshow(img_tinted.astype(np.uint8))

plt.show()