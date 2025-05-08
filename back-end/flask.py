from flask import Flask, request, send_file
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall
import io
import time
import base64  # 新增
from flask_cors import CORS  # 新增

# --- 自定义指标和层 ---
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def relu6(x):
    return K.relu(x, max_value=6)

class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.upsampling = upsampling
        self.output_size = output_size
        self.data_format = data_format

    def call(self, inputs):
        if self.output_size:
            return tf.image.resize(inputs, self.output_size, method='bilinear')
        else:
            new_size = [
                int(inputs.shape[1] * self.upsampling[0]),
                int(inputs.shape[2] * self.upsampling[1])
            ]
            return tf.image.resize(inputs, new_size, method='bilinear')

    def get_config(self):
        config = {
            'upsampling': self.upsampling,
            'output_size': self.output_size,
            'data_format': self.data_format
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

# --- 改进的可视化功能 ---
def generate_visualization(original_img, binary_mask):
    """生成原始图片叠加红色轮廓的效果图"""
    # 将掩码调整到原始图片尺寸
    resized_mask = cv2.resize(binary_mask, (original_img.shape[1], original_img.shape[0]))
    
    # 查找轮廓
    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建原图副本
    result_img = original_img.copy()
    
    # 绘制红色轮廓 (BGR格式)
    cv2.drawContours(result_img, contours, -1, (0, 0, 255), 2)
    
    # 添加半透明效果
    overlay = original_img.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), -1)  # 填充轮廓
    cv2.addWeighted(overlay, 0.2, result_img, 0.8, 0, result_img)
    
    return result_img

# --- Flask应用 ---
app = Flask(__name__)
CORS(app)  # 添加这一行
model = None



def load_model_simple():
    global model
    try:
        model = load_model(
            # './training_history/deeplab.hdf5',
            './training_history/2025-05-03 18-04-04.106775.hdf5',
            custom_objects={
                'relu6': relu6,
                'BilinearUpsampling': BilinearUpsampling,
                'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
                'dice_coef': dice_coef,
                'precision': Precision(),
                'recall': Recall()
            }
        )
        print("✅ 模型加载成功！输入尺寸:", model.input_shape)
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        raise e

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "未上传文件"}, 400
    
    file = request.files['file']
    if file.filename == '':
        return {"error": "空文件"}, 400

    try:
        # 读取原始图像（保留原始尺寸）
        img_bytes = file.read()
        original_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # 预处理预测用图像
        img_for_pred = cv2.resize(original_img.copy(), (224, 224))
        img_for_pred = img_for_pred / 255.0
        img_for_pred = np.expand_dims(img_for_pred, axis=0)

        # 预测
        pred = model.predict(img_for_pred, verbose=0)[0]
        binary_mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255

        # 获取输出类型参数（默认返回可视化结果）
        output_type = request.args.get('output', 'visualization')

        # 处理不同输出类型
        if output_type == 'visualization':
            result_img = generate_visualization(original_img, binary_mask)
            download_name = 'visualization.png'
        elif output_type == 'contour':
            contours, _ = cv2.findContours(
                cv2.resize(binary_mask, (original_img.shape[1], original_img.shape[0])),
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            contour_img = np.zeros(original_img.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_img, contours, -1, 255, 1)
            result_img = contour_img
            download_name = 'contour.png'
        elif output_type == 'mask':
            result_img = cv2.resize(binary_mask, (original_img.shape[1], original_img.shape[0]))
            download_name = 'mask.png'
        else:
            return {"error": "无效的输出类型"}, 400

        # 保存结果到服务器（带时间戳）
        #output_dir = os.path.join(os.path.dirname(__file__), "predictions")
        #os.makedirs(output_dir, exist_ok=True)
        #timestamp = time.strftime("%Y%m%d_%H%M%S")
        #output_path = os.path.join(output_dir, f"result_{timestamp}_{output_type}.png")
        #cv2.imwrite(output_path, result_img)
        #print(f"💾 结果已保存: {output_path}")

        # 返回图像响应
      #  _, img_png = cv2.imencode('.png', result_img)
  #      return send_file(
          #  io.BytesIO(img_png.tobytes()),
          #  mimetype='image/png',
          #  download_name=download_name
      #  )
   # 将结果图像转换为base64
        _, img_png = cv2.imencode('.png', result_img)
        img_base64 = base64.b64encode(img_png.tobytes()).decode('utf-8')
        
        return {
  "status": "success",
            "image": img_base64,
            "output_type": output_type
        }


    except Exception as e:
          return {"status": "error", "error": str(e)}, 500

@app.route('/test', methods=['GET'])
def test():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "output_modes": ["mask", "contour", "visualization"],
        "default_mode": "visualization"
    }

if __name__ == '__main__':
    load_model_simple()
    app.run(host='0.0.0.0', port=5000, debug=True)