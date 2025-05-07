import cv2
import numpy as np
import os
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from models.unets import Unet2D
from models.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D
from models.FCN import FCN_Vgg16_16s

from utils.learning.metrics import dice_coef, precision, recall
from utils.BilinearUpSampling import BilinearUpSampling2D
from utils.io.data import load_data, save_results, save_rgb_results, save_history, load_test_images, DataGen

# settings
input_dim_x = 224
input_dim_y = 224
color_space = 'rgb'
weight_file_name = 'deeplab.hdf5'
pred_save_path = '/root/wound-segmentation/data/Medetec_foot_ulcer_224/test/predictions/out/'

model = Deeplabv3(input_shape=(input_dim_x, input_dim_y, 3), classes=1)
model = load_model('./training_history/' + weight_file_name
               , custom_objects={'recall':recall,
                                 'precision':precision,
                                 'dice_coef': dice_coef,
                                 'relu6':relu6,
                                 'DepthwiseConv2D':DepthwiseConv2D,
                                 'BilinearUpsampling':BilinearUpsampling})

# Load the image you want to predict
image_path = '/root/wound-segmentation/data/Foot Ulcer Segmentation Challenge/test/images/1021.png'  


# 读取并预处理图片
image = cv2.imread(image_path)
image1 = cv2.resize(image, (input_dim_x, input_dim_y))  # 调整大小
image1 = image1 / 255.0  # 归一化
image1 = np.expand_dims(image1, axis=0)  # 添加 batch 维度

# 进行预测
prediction = model.predict(image1, verbose=1)

#将灰度图像二值化
#_, binary_image = cv2.threshold(prediction, 50, 255, cv2.THRESH_BINARY)

#转像素点数值类型
# prediction = (prediction * 255).astype(np.uint8)

# _, binary_image = cv2.threshold(prediction, 50, 255, cv2.THRESH_BINARY)

#prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)

# 确保保存路径存在
os.makedirs(pred_save_path, exist_ok=True)

# 保存预测结果
save_results(prediction, 'rgb', pred_save_path, [os.path.basename(image_path)])

#读取原图像
# image_bgr = cv2.imread('/root/wound-segmentation/data/Foot Ulcer Segmentation Challenge/test/images/1011.png')

# 检查图像是否成功加载.
if image is None:
    print("无法加载图像，请检查文件路径！")
else:
    # 将 BGR 转换为 RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"rgb_image尺寸: {rgb_image.shape}")

# 读取二值化图像
binary_image = cv2.imread('./data/Medetec_foot_ulcer_224/test/predictions/out/1021.png', cv2.IMREAD_GRAYSCALE)

# 查找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个空白图像用于绘制轮廓
boundary_image = np.zeros_like(binary_image)

# 绘制轮廓
cv2.drawContours(boundary_image, contours, -1, (255), thickness=1)

# 反转图像：255 - 当前像素值
boundary_image = 255 - boundary_image

print(f"boundary_image尺寸: {boundary_image.shape}")

# 调整灰度掩码大小以匹配 RGB 图像.
gray_mask_resized = cv2.resize(boundary_image, (rgb_image.shape[1], rgb_image.shape[0]))

print(f"gray_mask_resized尺寸: {gray_mask_resized.shape}")

# 将灰度掩码归一化到 [0, 1] 范围
#gray_mask_normalized = gray_mask_resized / 255.0

# 将灰度掩码扩展为 3 通道
#gray_mask_3_channels = cv2.merge([gray_mask_normalized, gray_mask_normalized, gray_mask_normalized])
gray_mask_3_channels = cv2.merge([gray_mask_resized, gray_mask_resized, gray_mask_resized])

print(f"gray_mask_3_channels尺寸: {gray_mask_3_channels.shape}")

#  掩码叠加
# blended_image = (rgb_image * gray_mask_3_channels).astype(np.uint8)

# 确保灰度掩码是 uint8 类型
if gray_mask_3_channels.dtype != np.uint8:
    gray_mask_3_channels = (gray_mask_3_channels * 255).astype(np.uint8)

# 确保 RGB 图像是 uint8 类型
if rgb_image.dtype != np.uint8:
    rgb_image = rgb_image.astype(np.uint8)

#cv2.imwrite('./data/Medetec_foot_ulcer_224/test/predictions/out/gray_mask_3_channels.jpg', gray_mask_3_channels)
#cv2.imwrite('./data/Medetec_foot_ulcer_224/test/predictions/out/rgb_image.jpg', rgb_image)
# 执行“或”操作
result = cv2.bitwise_and(rgb_image, gray_mask_3_channels)

result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
# 保存图像,最终要显示的图像
cv2.imwrite('./data/Medetec_foot_ulcer_224/test/predictions/out/output_image.jpg', result)
