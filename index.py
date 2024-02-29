

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ModelTrainingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ONPRICE INFOTECH Model Training")
        self.geometry("300x500")
        
        # Start Training Button
        self.start_button = ttk.Button(self, text="Start Training", command=self.prepare_and_start_training)
        self.start_button.pack(pady=20)
        
        # Frame for Matplotlib Figures
        self.fig_frame = tk.Frame(self)
        self.fig_frame.pack(fill=tk.BOTH, expand=True)

    def prepare_and_start_training(self):
        # Disable start button to prevent multiple clicks
        self.start_button["state"] = "disabled"
        try:
            # Prepare Data Generators
            data_path = 'C:/Users/DELL/Downloads/ONPRICE InfotechProject/Final Project/Groundnut_Leaf_dataset/copy of the dataset/Leaf Dataset ONPRICE INFOTECH'
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.2)  # Using 20% of the data for validation

            train_generator = train_datagen.flow_from_directory(
                data_path,
                target_size=(150, 150),
                batch_size=32,
                class_mode='categorical',
                subset='training')

            validation_generator = train_datagen.flow_from_directory(
                data_path,
                target_size=(150, 150),
                batch_size=32,
                class_mode='categorical',
                subset='validation')

            # Define Model
            model = self.build_model()

            # Train Model
            history = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // train_generator.batch_size,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // validation_generator.batch_size,
                epochs=10)

            # Plot Results
            self.plot_results(history)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            # Re-enable start button after training
            self.start_button["state"] = "normal"

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')  # Assuming 6 classes
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def plot_results(self, history):
        # Clear previous figures
        for widget in self.fig_frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot Accuracy
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='upper left')
        
        # Plot Loss
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='upper left')
        
        # Embedding Matplotlib Figure in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = ModelTrainingApp()
    app.mainloop()
