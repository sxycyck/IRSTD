#导出模型
python train/rtdetr_paddle/tools/export_model.py -c train/rtdetr_paddle/configs/rtdetr/rtdetr_focalnet_L_384_3x_coco.yml \
                             -o weights=../user_data/model_data/best_model.pdparams  trt=True \
                             --output_dir=../user_data/model_data/output_inference
                            
#转成onnx文件                    
paddle2onnx --model_dir=../user_data/model_data/output_inference/rtdetr_focalnet_L_384_3x_coco \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file ../user_data/model_data/rtdetr_focalnet_L_384_3x_coco.onnx
                            

