import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#Loading Saved models
model = load_model(r"C:\Users\jakem\OneDrive\Desktop\School\Machine Learning\Final Project\Saved_Model\PretrainedModel2.h5")
#model2 = load_model(r"C:\Users\jakem\OneDrive\Desktop\School\Machine Learning\Final Project\Saved_Model\CustomModel.h5")

#Defining Labels for Food in a Dictionary
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 
          3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 
          6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 
          9: 'corn', 10: 'cucumber', 11: 'eggplant', 
          12: 'garlic', 13: 'ginger',14: 'grapes', 
          15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 
          18: 'lettuce',19: 'mango', 20: 'onion', 
          21: 'orange', 22: 'paprika', 23: 'pear', 
          24: 'peas', 25: 'pineapple',26: 'pomegranate', 
          27: 'potato', 28: 'raddish', 29: 'soy beans', 
          30: 'spinach', 31: 'sweetcorn',32: 'sweetpotato', 
          33: 'tomato', 34: 'turnip', 35: 'watermelon', 36: 'dragon fruit'}

#Creating lists of Fruits and Vegetables the Model was trained on
fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon', 'dragon fruit']

vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

def scrape_for_calories(prediction) -> str:
    """
    This functions scrapes the web to find the calories of a passed in image
    """

    try:
        #Googling the calories in the passed in food, storing as a URL
        url_for_search = 'https://www.google.com/search?&q=calories in ' + prediction

        #Retrieving the text from the URL
        text = requests.get(url_for_search).text

        #Using the Beautiful Soup Web Scraper to parse the HTML and store in calories variable
        scraper = BeautifulSoup(text, 'html.parser')

        cal = scraper.find("div", class_="BNeawe iBp4i AP7Wnd").text

        return cal
    
    except Exception as error:

        #Returning error message from streamlit if it cannot find
        st.error("Was not able to detect the calories for this item, eat with caution!")
        print(error)

def return_facts(prediction):
    """
    This functions uses a huggingface model to return facts for the predicted food
    """
    #Checking if GPU compatiable device available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device

    #Loading in pretrained gpt 2 model from huggingface
    model_name = "gpt2-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #Transporting model to device
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    #Prompting model with fun facts about predicted food
    input_text = "Fun facts about " + prediction + "s"
    max_length = 110

    #Tokenizing input text
    input_ids = tokenizer(input_text, return_tensors="pt")

    input_ids = input_ids['input_ids'].to(device)

    #Generating model output with beam search
    output = model.generate(input_ids, max_length=max_length, num_beams=5,
                        do_sample=False, no_repeat_ngram_size=2)

    #Returning a decoded version (real words) of output
    return tokenizer.decode(output[0])

def return_recipe(prediction):
    """
    This function returns a webpage with recipes for the predicted food
    """
    text = f"Here's a great recipe using {prediction}s: https://www.allrecipes.com/search?q={prediction}"
    return text


def processed_img(img_path) -> str:
    """
    This function takes in the path of an image and returns the predicted label of the image
    using a pretrained model
    """
    #Loading image of the target size
    food_image = load_img(img_path, target_size=(224, 224, 3))

    #Converting image to an array of values
    food_image = img_to_array(food_image)

    #Normalizing the pixels in the image to between 0 and 1
    food_image = food_image / 255

    food_image = np.expand_dims(food_image, [0])

    #Predicting the image class using the given model
    pred = model.predict(food_image)
    
    #Storing answer as most probable prediction
    y_class = pred.argmax(axis=-1)

    #Printing the label from the list
    y = " ".join(str(x) for x in y_class)

    #Indexing the list of labels to retrieve the food 
    y = int(y)
    result = labels[y]

    #Returning label 
    return result.capitalize()


def run():
    """
    This function runs the streamlit server
    """
    #Defining title of the website
    st.title("Eat Right! Fruit and Vegetable Classification Model!")
    st.image(Image.open(r"C:\Users\jakem\FoodClassification\FOODCLASSIFICATION\upload_images\Untitled design (71).png"))

    #Allowing user to upload an image of their fruit or vegetable
    img_file = st.file_uploader("Upload your Ingredients!", type=["jpg", "png"])

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

        if img_file is not None:
            #Running prediction function on image if valid
            result = processed_img(save_image_path)

            #If the uploaded image is predicted to be a vegetable
            if result in vegetables:
                st.info('**This image is a vegetable!**')
            
            #If uploaded image predicted to be a fruit
            else:
                st.info('**This image is a fruit!**')

            #Using Streamlit success method
            st.success("**The model predicts this is a : " + result + '**')

            #Returning the calories of the predicted food label if valid
            cal_pred = scrape_for_calories(result)
            if cal_pred:
                st.warning('**' + cal_pred + ' for a serving of 100 grams**')

            recipe = return_recipe(result)
            if recipe:
                st.warning(recipe)
            
            #Returning some fun facts about the predicted food using the GPT-2 model
            facts = return_facts(result)
            if facts:
                st.warning('**Fun Facts: ' + facts + '!**')


if __name__ == '__main__':
    run()