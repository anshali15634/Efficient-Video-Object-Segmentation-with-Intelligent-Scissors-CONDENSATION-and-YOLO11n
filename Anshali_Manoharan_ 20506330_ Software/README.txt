# Individual Dissertation 2024-2025: Code Setup Instructions

Author: Anshali Manoharan


This README provides the steps necessary to set up and run the **Efficient Video Object Segmentation with Intelligent Scissors, CONDENSATION and CNN** project, including resolving common dependencies and errors related to OpenCV, Cython compilation, and OpenMP runtime issues.

## **Prerequisites**

Before starting, ensure you have the following installed:

- **Python 3.11.4**
- **Visual Studio Build Tools** (for compiling Cython extensions)
- **Microsoft C++ Build Tools** (for compiling extensions like `path_finding_cy_dynamic_training`)

Additionally, it is recommended to use Visual Studio Code Editor.

## **Step 1: Check File Directory**

Ensure the file directory of the submitted folder is as follows:
```
│   CONDENSATION.py
│   main.py
│   main_IS_Evaluation.py        
│   README.txt
│   setup_node.py
│   setup_path_finding.py        
│   yolo11n.pt
│   yolov11n-face.pt
│   YOLO_inference.py
│   __init__.py
│   
├───build
├───Evaluation_Images_Videos     
│       apple_cluttered_sample.mp4
│       billiards.mp4
│       boat.mp4
│       TED_Talk_facetracking.mp4
│
├───modules
│       image_processing.py
│       path_finding_cy_dynamic_training.pyx
│       path_finding_pure_python.py
│       __init__.py
│
└───utils
    │   node.pxd
    │   node.py
    │   node.pyx
    │   __init__.py
    │
    └───node
           Node.pxd

```
## **Step 2: Install Required Dependencies**
```
pip install opencv-python ultralytics numpy cython
```

## **Step 3: Compile Cython Extensions**

If you encounter an error like ImportError: cannot import name 'path_finding_cy_dynamic_training', you need to compile the Cython extension.

**Rebuild the Cython extension:** In the project directory, run the following command to compile the Cython extensions:
```
python setup_path_finding.py build_ext --inplace
python setup_node.py build_ext --inplace
```
## **Step 4: Re-check File Directory**

After compiling the Cython extensions, the file directory must look like this:

```
│   CONDENSATION.py
│   main.py
│   main_IS_Evaluation.py
│   README.md
│   setup_node.py
│   setup_path_finding.py
│   yolo11n.pt
│   yolov11n-face.pt
│   YOLO_inference.py
│   __init__.py
│
├───build
│   ├───lib.win-amd64-cpython-312
│   │   ├───modules
│   │   │       path_finding_cy_dynamic_training.cp312-win_amd64.pyd
│   │   │
│   │   └───utils
│   │           node.cp312-win_amd64.pyd
│   │
│   └───temp.win-amd64-cpython-312
│       └───Release
│           ├───modules
│           │       path_finding_cy_dynamic_training.cp312-win_amd64.exp
│           │       path_finding_cy_dynamic_training.cp312-win_amd64.lib
│           │       path_finding_cy_dynamic_training.obj
│           │
│           └───utils
│                   node.cp312-win_amd64.exp
│                   node.cp312-win_amd64.lib
│                   node.obj
│
├───Evaluation_Images_Videos
│       apple_cluttered_sample.mp4
│       billiards.mp4
│       boat.mp4
│       TED_Talk_facetracking.mp4
│
├───modules
│       image_processing.py
│       path_finding_cy_dynamic_training.c
│       path_finding_cy_dynamic_training.cp312-win_amd64.pyd
│       path_finding_cy_dynamic_training.pyx
│       path_finding_pure_python.py
│       __init__.py
│
└───utils
    │   node.c
    │   node.cp312-win_amd64.pyd
    │   node.pxd
    │   node.py
    │   node.pyx
    │   __init__.py
    │
    └───node
            Node.pxd
```
## **Step 5: Run**

Open `main.py` and run the script. A file picker should appear to select files for the first phase of the application, Intelligent Scissors.

**Note:** Any possible issues or errors faced after this is covered in the 'Possible Errors and Fixes' section at the bottom of the README, please refer to it.

## **Examples to try out**

In the Evaluation_Images_Videos folder, there are the 4 video sequences used in the dissertation. These can be picked to test the program. However, keep in mind that:
1. For the `TED_Talk_facetracking.mp4` video, the YOLO model needs to be changed in the main script. Please add the path `yolov11n-face.pt` in the parameter for the function `get_YOLO_model()` in lines 289 and 304.
2. For the given video sequences, the parameters `gaussian_diffusion` and `translation_range` need to be set according the parameters stated below:

| Video | gaussian_diffusion | translation_range |
|----------|----------|----------|
| apple_cluttered_sample   | 5  | 70  |
| billiards   | 7  | 150  |
| boat    | 7  | 150  |
| TED_Talk_facetracking    | 1  | 30  |

The parameters can be reset in lines 143 and 198 in `CONDENSATION.py`.

## **Possible Errors and Fixes**

### **Possible Error #1:**

If you get the following error:
```
cv2.error: OpenCV(4.x.x) error: (-2:Unspecified error) The function is not implemented...
```
### Solution
1. Uninstall existing OpenCV packages:
```
pip uninstall opencv-python opencv-python-headless
```

2. Reinstall OpenCV with GUI support:
```
pip install opencv-python
```

### **Possible Error #2:**

If you encounter OpenMP runtime issues such as:
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```
### Solution
1. Set the environment variable `KMP_DUPLICATE_LIB_OK=TRUE` to bypass the OpenMP runtime error:
```
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```
