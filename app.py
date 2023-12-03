import streamlit as st
import tensorflow as tf
from PIL import Image
from googleapiclient.discovery import build

# Load your trained model here
model = tf.keras.models.load_model('prowasterecycle.h5')

# Enter your API key here
api_key = "AIzaSyC5X8Xmwe1fZ-3nIV582rVQethKu6qn2KU"

# Define the API service
youtube = build('youtube', 'v3', developerKey=api_key)

# Define the search terms for each type of waste
search_terms = {
    "cardboard": "cardboard recycling",
    "glass": "glass recycling",
    "metal": "metal recycling",
    "paper": "paper recycling",
    "plastic": "plastic recycling",
    "trash": "trash waste"
}

# Define a function to preprocess the image and make predictions
def classify_image(image):
    # Preprocess the image
    image = image.resize((128, 128))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    # Make predictions with the model
    predictions = model.predict(image)

    # Get the class with the highest probability
    class_idx = tf.math.argmax(predictions, axis=1)[0]
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Replace with your own class names
    class_name = class_names[class_idx]

    return class_name

# Define a function to recommend videos based on the waste type
def recommend_videos(waste_type):
    # Search for videos related to the waste type
    search_query = search_terms.get(waste_type, "recycling")
    search_response = youtube.search().list(
        q=search_query,
        type='video',
        part='id,snippet'
    ).execute()

    # Return the search results as a list of video titles and IDs
    results = []
    for search_result in search_response.get('items', []):
        results.append({
            "title": search_result['snippet']['title'],
            "id": search_result['id']['videoId']
        })

    return results

# Define the Streamlit app
def app():
    st.title("Waste Classification and Video Recommendations")
    st.write("Upload an image to classify the type of waste and get video recommendations.")

    # Add file uploader to the app
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Classify the image and recommend videos when the user clicks the "Classify" button
    if uploaded_file is not None:
        # Load the image and preprocess it
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Classify the image and get the waste type
        waste_type = classify_image(image)

        # Recommend videos based on the waste type
        videos = recommend_videos(waste_type)

        # Display the waste type and recommended videos
        st.subheader("Waste Type: " + waste_type)
        st.subheader("Recommended Videos:")
        for video in videos:
            st.write("- " + video["title"])
            st.video("https://www.youtube.com/watch?v=" + video["id"])

if __name__ == "__main__":
    app()
