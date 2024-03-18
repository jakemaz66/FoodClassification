import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import re


def scrape_pixabay_images(search_query, num_images, output_folder):
    #Specifying the url of the site and desired images
    site = f'https://www.pexels.com/search/{search_query}/'

    #Sending get request to load site
    response = requests.get(site)

    #Parsing the html on the site, finding all images with tags
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')

    #Getting all image urls from the tages
    urls = [img['src'] for img in img_tags]

    #Stopping if exceeded specified image count
    count = 0
    for url in urls:
        if count >= num_images:
            break
        
        #If file is invalid with a non matching extension, added question mark for search parameter handling
        filename = re.search(r'/([\w_-]+\.(jpg|jpeg|gif|png))\?', url)
        if not filename:
            print("Regex didn't match with the url: {}".format(url))
            continue
        
        #Iterating count of images
        count += 1

        #Downloading image
        with open(os.path.join(output_folder, filename.group(1)), 'wb') as f:

            if 'http' not in url:

                url = urljoin(site, url)

            response = requests.get(url)
            f.write(response.content)

    print(f"Downloaded {count} images to {output_folder}")


if __name__ == '__main__':

    search_query = 'dragon fruit'
    num_images = 150
    output_folder = r'C:\Users\jakem\FoodClassification\FOODCLASSIFICATION\data'

    scrape_pixabay_images(search_query, num_images, output_folder)

    search_query = 'spinach'


    
