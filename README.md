# YOLO-V5 GRADCAM & GRADCAM++

I always wanted to know to which part of an object the object-detection model pays more attention. So I searched for it, but I didn't find any for Yolov5.
Here is my implementation of Grad-cam for YOLO-v5. If it benefited your project, please follow my GitHub account and give a ⭐️!


## Installation
`pip install -r requirements.txt`

## Infer
`python main.py --model-path yolov5.pt --img-path images/cat-dog.jpg`

**NOTE**: If you don't have any weights and want to test, Don't change the model-path variable. The yolov5s model will be automatically downloaded thanks to the download function from yolov5. 
## Examples
<img src="https://raw.githubusercontent.com/pooya-mohammadi/yolov5-gradcam/master/outputs/eagle-res.jpg" alt="cat&dog" height="300" width="1200">
<img src="https://raw.githubusercontent.com/pooya-mohammadi/yolov5-gradcam/master/outputs/cat-dog-res.jpg" alt="cat&dog" height="300" width="1200">
<img src="https://raw.githubusercontent.com/pooya-mohammadi/yolov5-gradcam/master/outputs/dog-res.jpg" alt="cat&dog" height="300" width="1200">

## Note
I checked the code, but I couldn't find an explanation for why the truck's heatmap does not show anything. Please inform me or create a pull request if you find the reason.



# References
1. https://github.com/1Konny/gradcam_plus_plus-pytorch
2. https://github.com/ultralytics/yolov5