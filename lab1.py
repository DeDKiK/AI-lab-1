import tkinter as tk
from tkinter import filedialog, Label, Button, Scale
import cv2
import numpy as np
from PIL import Image, ImageTk


class FeatureExtractionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Feature Extraction Tool")

        # UI components
        self.label = Label(master, text="Upload an Image:")
        self.label.pack()

        self.upload_button = Button(master, text="Choose File", command=self.upload_image)
        self.upload_button.pack()

        self.threshold_slider = Scale(master, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold")
        self.threshold_slider.pack()

        self.segments_slider = Scale(master, from_=2, to=10, orient=tk.HORIZONTAL, label="Number of Vertical Segments")
        self.segments_slider.pack()

        self.process_button = Button(master, text="Process Image", command=self.process_image)
        self.process_button.pack()

        self.image_label = Label(master)
        self.image_label.pack()

        self.features_label = Label(master, text="Feature Vectors:")
        self.features_label.pack()

    def upload_image(self):
        self.filepath = filedialog.askopenfilename()
        if self.filepath:
            self.display_image(self.filepath)

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((300, 300))
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_image)

    def process_image(self):
        # Load and preprocess the image
        img = cv2.imread(self.filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Error: Could not read the image. Check the file path or integrity.")
            return

        # Threshold the image
        _, thresholded = cv2.threshold(img, self.threshold_slider.get(), 255, cv2.THRESH_BINARY)

        # Segmentation and feature extraction
        segments = self.segments_slider.get()
        height, width = thresholded.shape
        segment_width = width // segments

        absolute_vector = []
        for i in range(segments):
            segment = thresholded[:, i * segment_width:(i + 1) * segment_width]
            count_black_pixels = np.sum(segment == 0)
            absolute_vector.append(count_black_pixels)

        # Normalization
        sum_normalized_vector = [x / sum(absolute_vector) for x in absolute_vector]
        max_normalized_vector = [x / max(absolute_vector) for x in absolute_vector]

        # Display feature vectors
        self.features_label.config(text=f"Absolute Vector: {[f'{x:.2f}' for x in absolute_vector]}\n" +
                                        f"Deresh S1: {[f'{x:.2f}' for x in sum_normalized_vector]}\n" +
                                        f"Deresh M1: {[f'{x:.2f}' for x in max_normalized_vector]}")

        # Display thresholded image with segment lines
        self.display_processed_image(thresholded, segments)

    def display_processed_image(self, img_array, segments):
        # Convert to color image for displaying segment lines
        img_color = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Draw vertical lines for each segment
        height, width = img_color.shape[:2]
        segment_width = width // segments
        for i in range(1, segments):
            x = i * segment_width
            cv2.line(img_color, (x, 0), (x, height), (0, 0, 255), 2)

        # Convert to PIL format for Tkinter
        img = Image.fromarray(img_color)
        img = img.resize((300, 300))
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_image)


root = tk.Tk()
app = FeatureExtractionApp(root)
root.mainloop()
