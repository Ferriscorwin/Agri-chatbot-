import streamlit as st
import ollama
from keras.preprocessing import image
from keras.models import load_model
import json
import tensorflow as tf
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image
import pandas as pd

xl_file = 'ulavar santhai.xlsx'
xl1_file='regulated.xlsx'
xl2_file='notified crops.xlsx'
with st.sidebar:
    # Create a dropdown list
    option = st.selectbox(
        'Select an option to know about market:',
        ('none','ulavar santhai', 'regulated market', 'Notified crops')
    )

    # Create a button
    button_clicked = st.button('Click me!')
    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"])

# Check if the button is clicked and show the selected option
if button_clicked:
    st.write('You selected:', option)

# Load the saved model

model= load_model('image_classification.h5')
clas = ["late blight", "early blight", "healthly"]

def predict_class(input_image, model):
    img = input_image.resize((150, 150))  # Resize the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image data

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    return predicted_class


# To display the xl file
def display_xl(file):
    if file is not None:
            df = pd.read_excel(file)
            st.write("### Uploaded Data:")
            st.dataframe(df)

            st.write("### Search Criteria:")
            search_column = st.selectbox("Select column to search:", df.columns)
            search_value = st.text_input("Enter search value:")

            if st.button("Search"):
                result_df = df[df[search_column] == search_value.capitalize()]
                st.write("### Search Result:")
                st.table(result_df)


st.title("💬 Agribot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

### Write Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

## Generator for Streaming Tokens
def generate_response():
    response = ollama.chat(model='llama2', stream=True, messages=st.session_state.messages)
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token
        yield token
def main():
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑‍💻").write(prompt)
        st.session_state["full_message"] = ""
        st.chat_message("assistant", avatar="🤖").write_stream(generate_response)
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})  
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(temp_image, caption='Uploaded Image.',width=300)

        # Predict class for the uploaded image
        predicted_class = predict_class(temp_image, model)
        print("Predicted class:", predicted_class)

        # Get the response based on the predicted class
        response = clas[predicted_class]
        if predicted_class==0 or predicted_class==1:
            st.write("our plant is affected by",response)
        else:
            st.write("Your plant is",response)
    #to give output for the option selected
    if option== 'ulavar santhai':
        display_xl(xl_file)
    if option== 'regulated market':
        display_xl(xl1_file)
    if option== 'Notified crops':
        display_xl(xl2_file)

if __name__ == "__main__":
    main()



