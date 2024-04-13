# Brain Tumor Detection App

Welcome to the Brain Tumor Detection App! This application utilizes deep learning to analyze MRI images for the presence of brain tumors. Built with Streamlit, TensorFlow, and OpenCV, it provides an easy-to-use web interface for uploading MRI images and instantly getting predictions.

## Live Application

Experience the application live: [Brain Tumor Detection App](https://braintumordetectionapp-deeplearning.streamlit.app/)

![Brain Tumor Detection App](screenshot.png)

## Features

- Upload MRI images in PNG, JPG, or JPEG format.
- Instantly get predictions on whether the uploaded MRI image indicates the presence of a brain tumor.
- View uploaded MRI images within the application sidebar.

## Installation

This application requires Python 3.11 and several dependencies, listed in `requirements.txt`.

To set up the local development environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Mrzeek/Brain_Tumor_Detection_App.git

    cd Brain_Tumor_Detection_App
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    streamlit run BrainTumorApp.py
    ```

## Usage

1. Click on the "Upload Brain MRI Image" button in the sidebar to upload an MRI image.
2. After uploading, you can click "Make Prediction" to process the image and view the prediction.
3. The application will display whether a brain tumor is detected or not.

## How It Works

### Image Preprocessing

The application preprocesses the MRI images by:

- Resizing the image to 240x240 pixels.
- Converting to grayscale and applying Gaussian blur.
- Performing thresholding to focus on relevant features.
- Cropping around the largest detected contour which is presumed to be the brain.

### Model Prediction

A pre-trained TensorFlow model predicts the presence of a brain tumor based on the preprocessed image.
The model's output is a probability score, which is interpreted to give a final prediction.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or create an issue if you have suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md)
 file for details.

## Contact

If you have any questions or comments about the app, please feel free to reach out.

GitHub: https://github.com/Mrzeek
