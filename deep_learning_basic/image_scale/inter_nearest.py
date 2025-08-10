import numpy as np
import cv2


def nearest(image, target_size):
    """
    最近邻插值缩放（支持放大和缩小）
    :param image: 输入图像 (H, W, 3)
    :param target_size: 目标尺寸 (new_h, new_w)
    :return: 缩放后的图像
    """
    target_h, target_w = target_size
    target_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # 缩放比例（原坐标到目标坐标）
    scale_h = image.shape[0] / target_h
    scale_w = image.shape[1] / target_w

    for tar_x in range(target_h):
        for tar_y in range(target_w):
            # 找到最近的原图像素坐标,round() 会把浮点数四舍五入成整数，选取最近的原图像素。
            src_x = min(int(round(tar_x * scale_h)), image.shape[0] - 1)
            src_y = min(int(round(tar_y * scale_w)), image.shape[1] - 1)

            # 赋值
            target_image[tar_x, tar_y] = image[src_x, src_y]

    return target_image


# 测试代码
if __name__ == "__main__":
    # 读取图片
    img = cv2.imread("cat.jpg")
    
    # 缩放到指定尺寸
    new_w, new_h = 1300, 1000
    target_img = nearest(img, (new_h, new_w))
    cv2.imshow("Nearest Neighbor Resize (Custom)", target_img)
      
    # 对比 OpenCV 内置方法
    # opencv_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow("Nearest Neighbor Resize (OpenCV)", opencv_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    