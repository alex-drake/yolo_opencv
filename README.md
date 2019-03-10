# YOLO with openCV
This repo shows how to use [YOLO](https://pjreddie.com/darknet/yolo/) detection with [OpenCV](https://opencv.org/), which is hella-simple now you can use the opencv function `cv2.dnn.readNetFromDarknet()`

## Set up
The repo is set up good to go except for the YOLO weights. For this code, we're using YOLO v3 so please make sure you download the correct weights from the YOLO page. You should be able to find them on [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights). Once downloaded, you should save the weights in the `yolo_cfg` folder.

## Images
To label images, please run `yolo_image.py`. If you find your images aren't being labelled then you may need to modify both the value for `threshold` and `min_confidence` - the current defaults for these are 0.3 and 0.5 respectively.

<img src="input/dog.jpg" alt="Goodboy somewhere" width="384" height="288" style="display:inline-block"/><img src="output/dog.png" alt="Goodboy there!" width="384" height="288" style="display:inline-block"/>

`yolo_image.py` will then detect all files in `input` with extension `'jpg','jpeg', 'bmp', 'png'`, and save the resulting detections in the `output` folder - these will be saved as `PNG`.

## Videos
To label a video, please run `yolo_video.py`. As with the above, you may need to change the threshold and condifence levels to suit your needs.

`yolo_video.py` will detect all files in `input` with extension `'mp4'` and save the resulting video to the `output` folder with extension `'mp4'`. 
Note that the code will only run detections on the first video file found (as otherwise it could run for a very long time).

## Caveats
If you are after something speedy then this is not for you - please consider the excellent [keras-yolo3](https://github.com/experiencor/keras-yolo3) from [Experiencor](https://github.com/experiencor) and consider running on a GPU rather than CPU.

## Thanks
This repo is based on the [tutorial](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/) from Adrian Rosebrock on PyImageSearch. There are loads of really good tutorials on that site so I heavily suggest that you try them out.