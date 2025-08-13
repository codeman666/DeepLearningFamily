import numpy as np

def cubic_weight(t, a=-0.5):
    t = abs(t)
    if t <= 1:
        return (a + 2) * t**3 - (a + 3) * t**2 + 1
    elif 1 < t < 2:
        return a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a
    else:
        return 0

def bicubic_interpolate_HWCN(image, out_h, out_w, align_corners=False):
    """
    image: numpy.ndarray, shape = [H, W, C]
    out_h, out_w: int, 目标图像高和宽
    align_corners: bool，是否角对齐
    返回: numpy.ndarray, shape = [out_h, out_w, C]
    """
    in_h, in_w, channels = image.shape
    output = np.zeros((out_h, out_w, channels), dtype=image.dtype)

    if align_corners:
        scale_x = (in_w - 1) / (out_w - 1) 
        scale_y = (in_h - 1) / (out_h - 1) 
    else:
        scale_x = in_w / out_w
        scale_y = in_h / out_h

    for y in range(out_h):
        for x in range(out_w):
            if align_corners:
                src_x = scale_x * x
                src_y = scale_y * y
            else:
                src_x = scale_x * (x + 0.5) - 0.5
                src_y = scale_y * (y + 0.5) - 0.5

            # 将计算出来的目标图像对应到原图的浮点坐标 
            # src_x, src_y 限制在合法范围内（0 到 图像宽/高减1）。
            src_x = min(max(src_x, 0), in_w - 1)
            src_y = min(max(src_y, 0), in_h - 1)

            # 即目标点相对于整数像素坐标的偏移。
            i = int(np.floor(src_x))
            j = int(np.floor(src_y))
            dx = src_x - i
            dy = src_y - j

            # 计算x方向和y方向的4个权重，这4个权重对应双三次插值邻域的4个像素权重
            # 水平方向的权重
            wx = np.array([cubic_weight(m - dx) for m in range(-1, 3)])
            # 垂直方向的权重
            wy = np.array([cubic_weight(dy - n) for n in range(-1, 3)])

            for c in range(channels):
                val = 0.0
                for m in range(-1, 3):
                    for n in range(-1, 3):
                        x_idx = min(max(i + m, 0), in_w - 1)
                        y_idx = min(max(j + n, 0), in_h - 1)
                        val += wx[m + 1] * wy[n + 1] * image[y_idx, x_idx, c]

                output[y, x, c] = val

    return output

if __name__ == "__main__":
    import cv2
    img = cv2.imread("F:\pytorchPorject\DeepLearningFamily\deep_learning_basic\image_scale\cat.jpg")  # HWC 格式
    if img is None:
        raise FileNotFoundError("图片路径错误或文件不存在")
    # 输出大小 300x400，注意cv2的shape是(H,W,C)
    out = bicubic_interpolate_HWCN(img, 300, 400, align_corners=False)
    out = np.clip(out, 0, 255).astype(np.uint8)  # 保证像素值合法
    cv2.imwrite("cat_bicubic.jpg", out)