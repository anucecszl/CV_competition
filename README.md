1. The codes apply new features in torch 1.9.0
2. The segment_cmp.py contains all the codes to train a torchvision segmentation model, there are three types of models are available: FCN resnet, deeplab v3, and lraspp_mobilenet_v3.
3. The segment_cmp_label.py will use the saved model to segment all the trianing images and save the RGB overlaid images.
4. segmentation_utils.py provides codes for IoU calculation and label visualisation
