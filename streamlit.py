import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

# Load your trained Keras model here
model = load_model('./model_food_101.h5')  # Ensure the model path is correct
print("Model loaded successfully")

# Define your class labels
class_names = [
    'Apple Pie', 'Baby Back Ribs', 'Baklava', 'Beef Carpaccio', 'Beef Tartare',
    'Beet Salad', 'Beignets', 'Bibimbap', 'Bread Pudding', 'Breakfast Burrito',
    'Bruschetta', 'Caesar Salad', 'Cannoli', 'Caprese Salad', 'Carrot Cake',
    'Ceviche', 'Cheesecake', 'Cheese Plate', 'Chicken Curry', 'Chicken Quesadilla',
    'Chicken Wings', 'Chocolate Cake', 'Chocolate Mousse', 'Churros', 'Clam Chowder',
    'Sandwich', 'Crab Cake', 'Creme Brulee', 'Croque Madame', 'Cupcakes',
    'Deviled Eggs', 'Donut', 'Dumplings', 'Edamame', 'Egg Benedict', 'Escargots',
    'Falafel', 'Filet Mignon', 'Fish Fry', 'Foie Gras', 'French Fries',
    'French Onion Soup', 'French Toast', 'Fried Calamari', 'Fried Rice',
    'Frozen Yogurt', 'Garlic Bread', 'Gnocchi', 'Greek Salad', 'Grilled Cheese Sandwich',
    'Grilled Salmon', 'Guacamole', 'Gyoza', 'Hamburger', 'Tomato Soup', 'Hotdog',
    'Huevos Rancheros', 'Hummus', 'Ice Cream', 'Lasagna', 'Lobster Bisque',
    'Lobster Roll Sandwich', 'Macaroni and Cheese', 'Macarons', 'Miso Soup', 'Mussels',
    'Nachos', 'Masala Omelette', 'Onion Rings', 'Oysters', 'Pad Thai', 'Paella', 'Pancake',
    'Panna Cotta', 'Peking Duck', 'Pho', 'Pizza', 'Pork Chop', 'Poutine', 'Prime Rib',
    'Pulled Pork Sandwich', 'Ramen', 'Ravioli', 'Red Velvet Cake', 'Risotto', 'Samosa',
    'Sashimi', 'Scallops', 'Seaweed Salad', 'Shrimp and Grits', 'Spaghetti Bolognese',
    'Spaghetti Carbonara', 'Spring Rolls', 'Steak', 'Strawberry Shortcake', 'Sushi',
    'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna Tartare', 'Waffles'
]


# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    try:
        st.write(f"Uploaded file type:{type(uploaded_file)}")
        st.write(f"Uploaded file type:{uploaded_file.size if hasattr(uploaded_file,'size') else 'Unkown'}")

        # Read the file content
        file_bytes = uploaded_file.read()

        # Debugging: Check if the file is read correctly
        if not file_bytes:
            raise ValueError("The uploaded file is empty or invalid.")

        st.write(f"File size: {len(file_bytes)} bytes")

        # Open the image from bytes and print debug information
        img = Image.open(BytesIO(file_bytes))

        # Print format of the image to verify it's being opened correctly
        st.write(f"Image format: {img.format}")
        #st.write(f"Image mode:{img.model}")
        st.write(f"Image size: {img.size}")

        # Convert to RGB (to handle different formats)
        img = img.convert('RGB')

        # Resize the image to the required dimensions (224x224)
        img = img.resize((224, 224))

        # Convert the image to a NumPy array
        img_array = np.array(img)

        # Normalize pixel values to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0

        # Add a batch dimension (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except UnidentifiedImageError:
        raise ValueError("The uploaded file is not a valid image.")
    except Exception as e:
        raise ValueError(f"An error occurred while processing the image: {e}")


# Streamlit App Header
st.header("NUTRICHEF")

# Title
st.title("Food Classification with NUTRICHEF")

# Image uploader
uploaded_file = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"])

# Display the uploaded image
#if uploaded_file is not None:
    #try:
        # Open the image using PIL
        #image = Image.open(uploaded_file)
        #st.image(image, caption="Uploaded Image", use_column_width=True)
        #st.write("Image uploaded successfully!")

    #except Exception as e:
        #st.error(f"Error displaying the image: {e}")

# Predict button
if st.sidebar.button("PREDICT"):
    if uploaded_file is not None:
        try:
            # Preprocess the image
            img_array = preprocess_image(uploaded_file)

            # Make predictions
            prediction = model.predict(img_array)

            # Get the predicted class index
            prediction_class_index = np.argmax(prediction)

            # Get the predicted class name
            predicted_class_name = class_names[prediction_class_index]

            # Display the result
            st.success(f"Predicted food item: **{predicted_class_name}**")
        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    else:
        st.warning("Please upload an image before making a prediction.")
