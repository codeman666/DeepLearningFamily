import numpy as np

def bilinear_interpolation(image, size, align_corners=True):
    """
    双线性插值 (H × W × C 格式)
    :param image: 输入图像，形状 (H, W, C)
    :param size: (target_h, target_w)
    :param align_corners: 是否对齐角点
    :return: 缩放后的图像
    """
    src_h, src_w, channels = image.shape
    target_h, target_w = size
    target_image = np.zeros((target_h, target_w, channels), dtype=np.uint8)

    if align_corners:
        scale_x = (src_w - 1) / (target_w - 1) 
        scale_y = (src_h - 1) / (target_h - 1) 
    else:
        scale_x = src_w / target_w
        scale_y = src_h / target_h

    for ty in range(target_h):
        for tx in range(target_w):
            if align_corners:
                src_x = tx * scale_x
                src_y = ty * scale_y
            else:
                src_x = (tx + 0.5) * scale_x - 0.5
                src_y = (ty + 0.5) * scale_y - 0.5

            # 限制坐标范围 np.floor() 是向下取整，比如 2.7 → 2，1.2 → 1。
            # x0、y0 就是映射位置的左上邻居像素索引。x1 = x0 + 1 表示右边的像素；y1 = y0 + 1 表示下边的像素。
            x0 = int(np.floor(src_x))
            y0 = int(np.floor(src_y))
            x1 = min(x0 + 1, src_w - 1)
            y1 = min(y0 + 1, src_h - 1)

            dx = src_x - x0
            dy = src_y - y0

            # 直接向量化取出四个邻点的 RGB
            top_left     = image[y0, x0]
            top_right    = image[y0, x1]
            bottom_left  = image[y1, x0]
            bottom_right = image[y1, x1]

            # 双线性插值公式 (对 RGB 三通道同时计算)
            top    = top_left * (1 - dx) + top_right * dx
            bottom = bottom_left * (1 - dx) + bottom_right * dx
            pixel  = top * (1 - dy) + bottom * dy

            target_image[ty, tx] = np.clip(pixel, 0, 255)

    return target_image

# 示例
if __name__ == "__main__":
    import cv2
    img = cv2.imread("F:\pytorchPorject\DeepLearningFamily\deep_learning_basic\image_scale\cat.jpg")  # HWC 格式
    out = bilinear_interpolation(img, (300, 400), align_corners=False)
    cv2.imwrite("cat_bilinear.jpg", out)
