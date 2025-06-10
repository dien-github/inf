# import numpy as np
# import tflite_runtime.interpreter as tflite

# class Model(object):
#     def __init__(self, model_path):
#         #super.__init__()
#         self.interpreter = tflite.Interpreter(model_path=model_path)
#         self.interpreter.allocate_tensors()

#         self.BOX_COORD_NUM = 4

#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()

#         # check the type of the input tensor
#         self.floating_model = self.input_details[0]["dtype"] == np.float32

#         # NxHxWxC, H:1, W:2
#         self.input_height = self.input_details[0]["shape"][1]
#         self.input_width = self.input_details[0]["shape"][2]

#         self.max_box_count = self.output_details[0]["shape"][2]

#         self.class_count = self.output_details[0]["shape"][1] - self.BOX_COORD_NUM
#         self.input_mean = 0.0
#         self.input_std = 255.0
#         self.keypoint_count = 0
#         self.score_threshold = 0.6

#     def prepare(self):
#         return None

#     def predict(self, image):
#         input_data = np.expand_dims(image, axis=0)

#         if self.floating_model:
#             input_data = (np.float32(input_data) - self.input_mean) / self.input_std

#         self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

#         self.interpreter.invoke()

#         output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
#         results = np.squeeze(output_data).transpose()

#         return results


import numpy as np

import tflite_runtime.interpreter as tflite
# import tensorflow as tf

# tflite = tf.lite
# import tflite_micro_runtime.interpreter as tflite


class Model(object):
    def __init__(self, model_path):
        # initial interpreter
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # get input/output tensor
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        self.floating_model = self.input_details[0]["dtype"] == "float32"

        # Expecting input shape [1, channels, height, width]
        self.input_height = self.input_details[0]["shape"][2]
        self.input_width = self.input_details[0]["shape"][3]
        self.input_channels = self.input_details[0]["shape"][1]

        self.num_classes = self.output_details[0]["shape"][1]

        self.input_mean = 0.0
        self.input_std = 255.0

        self.keypoint_count = 0
        self.score_threshold = 0.6

    def prepare(self):
        return None

    def predict(self, image):
        # Đảm bảo image là numpy array RGB
        if not hasattr(image, "shape"):
            image = np.array(image.convert("RGB"))  # Đảm bảo 3 kênh

        # Nếu ảnh là (H, W, 3), chuyển về (3, H, W)
        if image.shape == (self.input_height, self.input_width, self.input_channels):
            image = np.transpose(image, (2, 0, 1))
        # Nếu ảnh là (H, W), thêm chiều channel
        elif image.shape == (self.input_height, self.input_width):
            image = np.expand_dims(image, axis=0)

        # Đảm bảo shape đúng (3, H, W)
        if image.shape != (self.input_channels, self.input_height, self.input_width):
            raise ValueError(
                f"Input image shape {image.shape} does not match model expected shape ({self.input_channels}, {self.input_height}, {self.input_width})"
            )

        input_data = np.expand_dims(image, axis=0)  # (1, 3, H, W)

        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
        else:
            input_data = input_data.astype(self.input_details[0]["dtype"])

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        return np.squeeze(output_data)
