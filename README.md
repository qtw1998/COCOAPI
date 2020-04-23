# COCOAPI

>  take YOLOv3 as instance

## 1 Waiting List

- [x] Restructured the COCO API Remember download `pycocotools` filefolder at the same time.

    > `cocoMetrix.py`

- [ ] Upload the Conversion Codes

    - [ ] BDD100K(JSON) → VOC(XML & TXT) 
    - [ ] VOC(XML & TXT) →COCO(JSON-integrated)
    - [ ] TFRecord & validation-JSON

## 2 Usage

### 2.1 COCO-API criteria

#### 2.1.1 Default settings

![image-20200423205351356](https://site-pictures.oss-eu-west-1.aliyuncs.com/q8onq.png)

#### 2.1.2 Testing Results on BDD100K (YOLOv3-SPP3)

```bash
(xxxx) [xxxxx@head1 yolov3]$ CUDA_VISIBLE_DEVICES=3,4,5,6,7 python test.py
True
Namespace(augment=False, batch_size=120, cfg='cfg/yolov3-spp3.cfg', conf_thres=0.001, data='data/bdd100k.data', device='', img_size=640, iou_thres=0.7, save_json=False, single_cls=False, task='test', weights='/cluster/home/qiaotianwei/yolo/yolov33/bdd100k_yolov3-spp3_final.weights')
True
Using CUDA device0 _CudaDeviceProperties(name='GeForce RTX 2080 Ti', total_memory=11019MB)
           device1 _CudaDeviceProperties(name='GeForce RTX 2080 Ti', total_memory=11019MB)
           device2 _CudaDeviceProperties(name='GeForce RTX 2080 Ti', total_memory=11019MB)
           device3 _CudaDeviceProperties(name='GeForce RTX 2080 Ti', total_memory=11019MB)
           device4 _CudaDeviceProperties(name='GeForce RTX 2080 Ti', total_memory=11019MB)

cuda:0
Model Summary: 225 layers, 6.38998e+07 parameters, 6.38998e+07 gradients
Fusing layers...
Model Summary: 152 layers, 6.38729e+07 parameters, 6.38729e+07 gradients
Reading image shapes: 100%|███████████████████████████████████████████████████████████| 9999/9999 [00:00<00:00, 13994.62it/s]
Caching labels (9999 found, 0 missing, 0 empty, 0 duplicate, for 9999 images): 100%|███| 9999/9999 [00:01<00:00, 5105.47it/s]
loading annotations into memory...
Done (t=1.37s)
creating index...
index created!
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|██████| 84/84 [05:21<00:00,  3.82s/it]
                 all     1e+04  1.86e+05     0.313     0.494     0.399     0.382
                 car     1e+04  1.02e+05     0.441     0.758     0.687     0.557
                 bus     1e+04   1.6e+03     0.399     0.546     0.495     0.461
              person     1e+04  1.33e+04     0.321     0.578     0.449     0.413
                bike     1e+04  1.01e+03     0.277     0.429     0.304     0.336
               truck     1e+04  4.24e+03       0.4      0.58     0.508     0.473
               motor     1e+04       452     0.283     0.358     0.252     0.316
               train     1e+04        15         0         0         0         0
               rider     1e+04       649     0.286      0.41     0.305     0.337
        traffic sign     1e+04  3.49e+04     0.377     0.652     0.543     0.477
       traffic light     1e+04  2.69e+04     0.348     0.628     0.444     0.448
Speed: 5.8/2.3/8.0 ms inference/NMS/total per 640x640 image at batch-size 120
creating index...
index created!
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.172
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.138
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.055
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.233
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.147
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.192
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.538
```
