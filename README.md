# Computer Vision Food Recognition and Nutritional Analysis

## Overview
This project aims to develop a computer vision application that can scan images, recognize and label the food in the image, provide their nutritional profiles, suggest recipes based on the identified ingredients, and implement a chatbot that can generate text about the food based on user input. The application is hosted at https://jakemazmachinelearning.streamlit.app/ with a demo at https://www.youtube.com/watch?v=a_KT_N2hFzY&feature=youtu.be, allowing users to upload images for analysis. 

## Technologies Used
- Python
- Streamlit
- TensorFlow
- Kaggle Food Recognition Challenge Dataset
- Beautiful Soup Web Scraper
- HuggingFace 

## Deep Learning Model
The model uses the MobileNet_V2 architecture to identify food within the images. This was the best perfomring model out of 3 attempted ones. The model was trained on 50 epochs of the collected training data and achieved an accuracy of over 98%. If the confidence of the model is below 50% for any given prediction, it will return an error indicating that it cannot recognize the food within the image.

## Data
Data for the model was achived via webscraping by Beautiful soup for images of fruits and vegetables, but also augemented using the Fruits and Vegetables Image Recognition Dataset (kaggle.com)
(https://github.com/jakemaz66/FoodClassification/assets/133889822/75915eaf-dc9f-4213-bb04-e22b371220ad)

## Nutritional Information
Nutritional information for identified food items is returned via a search query from nutrionix.com and reuturned in a streamlit text field as a hyperlink

## Recipe Suggestions
Relvant Recipes for the identified food items is returned via a search query from allrecipes.com and reuturned in a streamlit text field as a hyperlink

## GPT-2 Model
A Chatbot downloaded from the pre-trained GPT-2 model from OpenAI was employed to answer questions about the food and generate text from user input. The model uses a combination of beam search and top-k sampling decoding in order to generate text. This model is still being developed and fine-tuned on custom food-related data, so in its current state it is subject to frequent hallucinations.

## Streamlit
The app is hosted on Streamlit, an application that allows users to deisgn web applications. The website contains the app and a documetation page. The website is custom themed through the .streamlit/config.toml file, and contains all the functionality of the project desribed previously.

## Using the APP
A Documentation of the app can be found at the "Docs" page here: https://jakemazmachinelearning.streamlit.app/

## Future Enhancements
- Improve model accuracy and performance through iterative training and fine-tuning.
- Enhance user interface design and experience for a more polished application.
- Expand nutritional analysis capabilities to include more detailed metrics and insights.
- Incorporate additional datasets or APIs for broader food recognition and nutritional information coverage.
- Introduce larger and better language models to produce more useful generated text from user inputs
