import streamlit as st
from PIL import Image
import re
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM)


#Loading Saved models
model = load_model("app/PretrainedModel2.h5")
#model2 = load_model(r"C:\Users\jakem\OneDrive\Desktop\School\Machine Learning\Final Project\Saved_Model\CustomModel.h5")

#Defining Labels for Food in a Dictionary
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 
          3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 
          6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 
          9: 'corn', 10: 'cucumber', 11: 'dragon fruit', 12: 'eggplant', 
          13: 'garlic', 14: 'ginger',15: 'grapes', 
          16: 'jalepeno', 17: 'kiwi', 18: 'lemon', 
          19: 'lettuce',20: 'mango', 21: 'onion', 
          22: 'orange', 23: 'paprika', 24: 'pear', 
          25: 'peas', 26: 'pineapple',27: 'pomegranate', 
          28: 'potato', 29: 'raddish', 30: 'soy beans', 
          31: 'spinach', 32: 'sweetcorn',33: 'sweetpotato', 
          34: 'tomato', 35: 'turnip', 36: 'watermelon'}

#Creating lists of Fruits and Vegetables the Model was trained on to predict food category
fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Dragon Fruit', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']

vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

def scrape_for_calories(prediction) -> str:
    """
    This functions scrapes the web to find the calories of tge predicted food 
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
    

def scrape_for_protein(prediction) -> str:
    """
    This functions scrapes the web to find the protein of a passed in image
    """
    try:
        #Googling the calories in the passed in food, storing as a URL
        url_for_search_2 = 'https://www.google.com/search?&q=protein in ' + prediction

        #Retrieving the text from the URL
        text2 = requests.get(url_for_search_2).text

        #Using the Beautiful Soup Web Scraper to parse the HTML and store in calories variable
        scraper = BeautifulSoup(text2, 'html.parser')
        protein = scraper.find("div", class_="BNeawe iBp4i AP7Wnd").text

        return protein

    except Exception as error:
        #Returning error message from streamlit if it cannot find
        st.error("Was not able to detect the protein for this item, eat with caution!")
        print(error)



def generate_text(input_prompt):
    """
    This functions uses a huggingface model to generate conversational text about the predicted food
    """
    #Checking if GPU compatiable device available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Loading in pretrained gpt 2 model from huggingface
    model_name = "gpt2-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #Transporting model to device
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    #Prompting model with fun facts about predicted food
    input_text = input_prompt
    #force_words = [prediction]
    max_length = 100

    #Tokenizing input text
    input_ids = tokenizer(input_text, return_tensors="pt")
    #force_words_ids = tokenizer(force_words,return_tensors="pt")

    input_ids = input_ids['input_ids'].to(device)
    #force_words_ids = force_words_ids['force_words_ids'].to(device)

    #Generating model output with beam search and top k sampling
    output = model.generate(input_ids,
                            #force_words_ids=force_words_ids,
                            max_length=max_length, 
                            num_beams=10,
                            num_return_sequences=10, 
                            no_repeat_ngram_size=2,
                            do_sample=True,
                            top_k=50,
                            early_stopping=True
    )

    #Returning a decoded version (real words) of output
    output =  tokenizer.decode(output[0])

    #Ensuring that the final output ends at a punctuation mark
    sentences = re.split(r'(?<=[.!?]) +', output)
    final_output = ""
    for sentence in sentences:
        if sentence.strip() and any(c in sentence for c in (".", "!", "?")):
            final_output += sentence + " "

    return final_output


def return_recipe(prediction):
    """
    This function returns a webpage with recipes for the predicted food
    """
    text = f"Here's some great recipes using {prediction}s: https://www.allrecipes.com/search?q={prediction}"
    return text

def return_nutrition(prediction):
    """
    This function returns a webpage with nutrition for the predicted food
    """
    text = f"Here's the complete nutrition breakdown of {prediction}s: https://www.nutritionix.com/search?q={prediction}"
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

    #Getting confidence of prediction
    max_confidence = pred.max()

    #Printing the label from the list
    y = " ".join(str(x) for x in y_class)

    #Indexing the list of labels to retrieve the food 
    y = int(y)
    result = labels[y]

    #Setting a threshold for food confidence to reject non-food images 
    if max_confidence < 0.5:
        return "I'm sorry, I don't recognize this as a food"

    #Returning label 
    return result.capitalize()


def run():
    """
    This function runs the streamlit server
    """
    # Create a sidebar menu
    menu = st.sidebar.radio('Navigation', ['Home', 'Docs'])

    if menu == 'Home':

        #Defining title of the website
        st.title("Eat Right! Fruit and Vegetable Classification Model!")
        st.image(Image.open('app/upload_images/Untitled design - 2024-04-01T065557.621.png'), caption='We use neural nets to power this app!')

        #Allowing user to upload an image of their fruit or vegetable
        img_file = st.file_uploader("Upload your Ingredients!", type=["jpg", "png"])

        if img_file is not None:
            #Reading in the image
            img = Image.open(img_file).resize((250, 250))

            #Displaying the image in streamlit app
            st.image(img, use_column_width=False)

            #Saving file path of uploaded image
            save_image_path = 'app/upload_images/' + img_file.name 

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
                elif result in fruits:
                    st.info('**This image is a fruit!**')

                else:
                    st.info('I cannot recognize this food')

                #Using Streamlit success method
                if result in fruits or result in vegetables:
                    st.success("**The model predicts this is a : " + result + '**')

                    #Returning the calories of the predicted food label if valid
                    cal_pred = scrape_for_calories(result)
                    if cal_pred:
                        st.warning('**' + cal_pred + ' for a serving of 100 grams**')
                    
                    #Returning the protein of the predicted food label if valid
                    protein_pred = scrape_for_protein(result)
                    if cal_pred:
                        st.warning('**' + protein_pred + ' of protein for a serving of 100 grams**')

                    #Returning the nutrition breakdown
                    if result in fruits or result in vegetables:
                        nutrition = return_nutrition(result)
                        if nutrition:
                            st.warning(nutrition)
                    else:
                        st.warning('Cannot find nutrition breakdown for this food')

                elif result not in fruits or result not in vegetables:
                        st.warning('The Model cannot find information for this food')

                #Returning the recipes
                if result in fruits or result in vegetables:
                    recipe = return_recipe(result)
                    if recipe:
                        st.warning(recipe)
                else:
                    st.warning('Cannot find recipes for this food')      


                st.warning("This language model may be subject to incorrect output, please consider its output with caution")
                if result in fruits or result in vegetables:
                    huggingface_input = st.text_input("Start a Conversation about " + result +"!")

                    length = True
                    
                    #Putting constraint on length of input text
                    if len(huggingface_input) > 50:
                        st.warning('Prompts limited to 50 characters, Try a Shorter Prompt!')
                        length = False

                    elif huggingface_input and length == True:
                        text = generate_text(huggingface_input)
                        st.warning(text)

    else:
        #Defining the Documentation Page

        st.title('How to Use my App')
        st.markdown("""
        ## How the App Works

        This app can help you make healthier food choices by giving you the nutrition content, relevant recipes, 
        and information about ingredients you have on hand, all from a simple photo! Here's some functions of the app:

        1. First, take a picture of a fruit or vegetable you have. Make sure the photo is high resolution and the ingredient is the
           primary focus of the image. The model currently supports 37 different fruits and vegetables. If your ingredient is
           not recognized, the model will display a message informing that it cannot recognize the food.
        2. Next, upload your image using the upload button within the application. Images should be in png or jpeg format.
        3. Then, Our fine-tuned neural network will predict the fruit/vegetable from your image and classify it into a category.
        4. Our app will then return the calories for a 100 gram serving of the prediction, and provide a link to recipes
           at https://www.allrecipes.com/
        5. Finally, using a GPT-2 model fine-tuned with beam-search and top-k sampling, you can ask
           questions about your food to a chatbot. Good questions to ask are things like the 
           vitamin breakdown in the food, creative ways to use it in cooking, or fun facts about the ingredient. 
           NOTE: Prompts to the chatbot are currently limited to 50 characters or less! The responses from the chatbot
                 may be subject to non-factual remarks as well as it is still a small model, so please proceed with caution!


        Please enjoy my food classification app!
                    
        For any questions you can contact me here: Mazurkiewiczj@duq.edu
        """)
    

if __name__ == '__main__':
    run()