import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

#Loading Saved model
model = load_model(r"C:\Users\jakem\OneDrive\Desktop\School\Machine Learning\Final Project\Saved_Model\PretrainedModel2.h5")

#Defining Labels for Food
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

#Creating lists of Fruits and Vegetables
fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']

vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']


def fetch_calories(prediction) -> str:
    """
    This functions scrapes the web to find the calories of a passed in image
    """

    try:
        #Googling the calories in the passed in food, storing as a URL
        url = 'https://www.google.com/search?&q=calories in ' + prediction

        #Retrieving the text from the URL
        req = requests.get(url).text

        #Using the Beautiful Soup Web Scraper to parse the HTML and store in calories variable
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text

        return calories
    
    except Exception as e:
        #Returning error message from streamlit if it cannot find
        st.error("Can't able to fetch the Calories")
        print(e)


def processed_img(img_path):
    """
    This function takes in the path of an image and returns the predicted label of the image
    using a pretrained model
    """

    #Loading image of the target size
    img = load_img(img_path, target_size=(224, 224, 3))

    #Converting image to an array of values
    img = img_to_array(img)

    #Normalizing the pixels
    img = img / 255

    img = np.expand_dims(img, [0])

    #Predicting the image class using the given model
    answer = model.predict(img)
    
    #Storing answer as most probable prediction
    y_class = answer.argmax(axis=-1)

    print(y_class)

    #Printing the label from the list
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    #Returning label 
    return res.capitalize()


def run():
    #Defining title
    st.title("Eat Right! Fruit and Vegetable Classification Model!")

    #Allowing user to upload an image of their fruit or vegetable
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])

    if img_file is not None:
        #Reading in the image
        img = Image.open(img_file).resize((250, 250))

        #Displaying the image in streamlit app
        st.image(img, use_column_width=False)

        #Saving file path of uploaded image
        save_image_path = r'C:\Users\jakem\FoodClassification\FOODCLASSIFICATION\upload_images' + img_file.name 

        #Saving image to new path
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            #Running prediction function on image if valid
            result = processed_img(save_image_path)
            print(result)

            #If the uploaded image is predicted to be a vegetable
            if result in vegetables:
                st.info('**Category : Vegetables**')
            
            #If uploaded image predicted to be a fruit
            else:
                st.info('**Category : Fruit**')

            #Using Streamlit success method
            st.success("**Predicted : " + result + '**')

            #Returning the calories of the predicted food label if valid
            cal = fetch_calories(result)
            if cal:
                st.warning('**' + cal + '(100 grams)**')


if __name__ == '__main__':
    run()