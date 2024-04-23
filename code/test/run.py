import numpy as np
from onnxruntime import InferenceSession
import os
import cv2
import onnxruntime as onnxrt

root = '/work/data/object-detection-for-tilting-ships-test-set'
dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
model_path = dir+'/user_data/model_data/rtdetr_focalnet_L_384_3x_coco.onnx'

model = InferenceSession(model_path,providers=['CUDAExecutionProvider'])
output_name = model.get_outputs()[0].name
input_name = model.get_inputs()
def predict(img_path):
    img = cv2.imread(img_path)
    h,w,_=img.shape
    s1 = 640.0/h
    s2 = 640.0/w
    img = cv2.resize(img,(640,640),interpolation=cv2.INTER_CUBIC).astype('float32')
    img = np.swapaxes(img,0,2)
    img = np.swapaxes(img,1,2)
    img = np.expand_dims(img,axis=0)
    img = img/255.0
    img = onnxrt.OrtValue.ortvalue_from_numpy(img, 'cuda', 0)
    im_shape = onnxrt.OrtValue.ortvalue_from_numpy(np.asarray([[640,640]]).astype('float32'),'cuda',0)
    scale_factor = onnxrt.OrtValue.ortvalue_from_numpy(np.asarray([[s1,s2]]).astype('float32'),'cuda',0)
    output = np.zeros((300,6)).astype('float32')
    output = onnxrt.OrtValue.ortvalue_from_numpy(output,'cuda',0)
    output_name = model.get_outputs()[0].name
    input_name = model.get_inputs()
    io_binding = model.io_binding()
    io_binding.bind_ortvalue_input(name =input_name[0].name,ortvalue = im_shape)
    io_binding.bind_ortvalue_input(name =input_name[1].name,ortvalue = img)
    io_binding.bind_ortvalue_input(name =input_name[2].name,ortvalue = scale_factor)
    io_binding.bind_ortvalue_output(name =output_name,ortvalue=output)
    model.run_with_iobinding(io_binding)
    z = io_binding.get_outputs()
    return z[0].numpy(),h,w

files = os.listdir(root)
for file in files:
    image_path = root+'/'+file
    out,h,w = predict(image_path)
    dst = dir+'/prediction_result/output'
    if not os.path.exists(dst):
        os.mkdir(dst)
    txt_path = dst+'/'+file[:-3]+'txt'
    with open(txt_path,'w') as f:
        for i in range(len(out)):
            l = out[i].tolist()
            box = l[2:]
            for i in range(4):
                box[i] = int(box[i]+0.5)
                box[i] = max(0,box[i])
                if i%2:
                    box[i]=min(h,box[i])
                else:
                    box[i]=min(w,box[i])
                
            f.write(f'ship {l[1]} {box[0]} {box[1]} {box[2]} {box[3]}\n')


    