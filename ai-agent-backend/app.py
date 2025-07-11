from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import spacy
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
from bs4 import BeautifulSoup
from forex_python.converter import CurrencyRates
import speech_recognition as sr

# Set up OpenAI API for Generative AI
openai.api_key = 'your-openai-api-key'

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

# User preferences
user_preferences = {
    "price_range": (50, 500),
    "preferred_brands": ["BrandA", "BrandB"],
    "keywords": ["laptop", "gaming"],
    "sort_by": "price",  # Options: price, rating, popularity
}

app = Flask(__name__)
CORS(app)

# Function to initialize Selenium WebDriver
def init_driver():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.maximize_window()
    return driver

# Function to scrape product data using Selenium
def scrape_product_data_selenium(url):
    driver = init_driver()
    driver.get(url)
    time.sleep(3)  # Allow the page to load

    products = []
    try:
        product_elements = driver.find_elements(By.CSS_SELECTOR, '.s-item')
        for product in product_elements:
            name = product.find_element(By.CSS_SELECTOR, '.s-item__title').text
            price = product.find_element(By.CSS_SELECTOR, '.s-item__price').text
            link = product.find_element(By.CSS_SELECTOR, '.s-item__link').get_attribute('href')
            products.append({"name": name, "price": price, "url": link})
    except Exception as e:
        print(f"Error scraping data: {e}")
    finally:
        driver.quit()

    return products

# Function to fetch product data from dynamic websites using Selenium
def fetch_dynamic_product_data(product_type, site):
    if site == "ebay":
        url = f'https://www.ebay.com/sch/i.html?_nkw={product_type}'
    elif site == "walmart":
        url = f'https://www.walmart.com/search/?query={product_type}'
    else:
        return []

    return scrape_product_data_selenium(url)

# Extract keywords from the product type using NLP
def extract_keywords(text):
    doc = nlp(text)
    keywords = []
    for token in doc:
        if token.pos_ in ['NOUN', 'ADJ']:  # Extract nouns and adjectives
            keywords.append(token.text.lower())
    return " ".join(keywords)

# Function to scrape product data from websites
def scrape_product_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Example: Extract product details
        products = []
        for product in soup.select('.product-card'):
            name = product.select_one('.product-title').text
            price = float(product.select_one('.product-price').text.replace('$', ''))
            rating = float(product.select_one('.product-rating').text)
            products.append({"name": name, "price": price, "rating": rating})
        return products
    else:
        print(f"Error scraping {url}: {response.status_code}")
        return []

# Function to analyze product reviews using NLP
def analyze_reviews(reviews):
    prompt = f"Analyze the sentiment and key features of the following reviews: {reviews}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function for dynamic price comparison
def compare_prices(products):
    sorted_products = sorted(products, key=lambda x: x[user_preferences['sort_by']])
    print("Top products based on your preferences:")
    for product in sorted_products[:5]:
        print(f"Name: {product['name']}, Price: ${product['price']}, Rating: {product['rating']}")

# Scrape products from eBay or Walmart based on the product type
def scrape_products(product_type, site):
    if site == "ebay":
        url = f'https://www.ebay.com/sch/i.html?_nkw={product_type}'
    elif site == "walmart":
        url = f'https://www.walmart.com/search/?query={product_type}'
    else:
        return []

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the page from {site}.")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    products = []

    if site == "ebay":
        listings = soup.select('.s-item')
        for listing in listings:
            name = listing.select_one('.s-item__title')
            price = listing.select_one('.s-item__price')
            link = listing.select_one('.s-item__link')
            if name and price and link:
                try:
                    price_value = float(price.text.strip().replace('$', '').replace(',', ''))
                    products.append({
                        "name": name.text.strip(),
                        "price": price_value,
                        "url": link['href']
                    })
                except ValueError:
                    continue

    elif site == "walmart":
        listings = soup.select('.search-result-gridview-item')
        for listing in listings:
            name = listing.select_one('.product-title-link')
            price = listing.select_one('.price-main .visuallyhidden')
            link = listing.select_one('.product-title-link')
            if name and price and link:
                try:
                    price_value = float(price.text.strip().replace('$', '').replace(',', ''))
                    products.append({
                        "name": name.text.strip(),
                        "price": price_value,
                        "url": f"https://www.walmart.com{link['href']}"
                    })
                except ValueError:
                    continue

    return products

# Convert currency using forex-python
def convert_currency(amount, from_currency, to_currency):
    cr = CurrencyRates()
    return cr.convert(from_currency, to_currency, amount)

# Voice command recognition function
def recognize_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            command = recognizer.recognize_google(audio)
            print(f"Command recognized: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError:
            print("Sorry, the speech recognition service is unavailable.")
            return None

# API endpoint to get product recommendations
@app.route("/api/auth/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        budget = float(data.get("budget", 0))
        product_type = data.get("product", "").lower()
        use_nlp = data.get("use_nlp", False)
        currency = data.get("currency", "USD")
        language = data.get("language", "en")

        if use_nlp:
            product_type = extract_keywords(product_type)  # Process the product type with NLP

        ebay_products = scrape_products(product_type, "ebay")
        walmart_products = scrape_products(product_type, "walmart")

        all_products = ebay_products + walmart_products
        recommendations = [
            product for product in all_products if product["price"] <= budget
        ]

        # Convert prices based on the selected currency
        for product in recommendations:
            product["price"] = convert_currency(product["price"], "USD", currency)

        recommendations = sorted(recommendations, key=lambda x: x['price'])

        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# API endpoint for voice search
@app.route("/api/auth/voice_search", methods=["GET"])
def voice_search():
    try:
        command = recognize_voice_command()
        if command:
            # Use the voice command as the product type
            data = {
                "product": command,
                "budget": 100,  # Default budget
                "use_nlp": True,  # Use NLP for processing
                "currency": "USD",
                "language": "en"
            }
            # Call the recommend endpoint with the voice command
            response = recommend()
            return response
        else:
            return jsonify({"error": "No valid voice command recognized."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Main function for fetching and comparing products dynamically
@app.route("/api/auth/fetch_and_compare", methods=["GET"])
def fetch_and_compare():
    try:
        api_urls = [
            "https://api.ecommerce1.com/products",
            "https://api.ecommerce2.com/products"
        ]
        params = {
            "keywords": " ".join(user_preferences["keywords"]),
            "min_price": user_preferences["price_range"][0],
            "max_price": user_preferences["price_range"][1]
        }

        all_products = []
        for api_url in api_urls:
            all_products.extend(fetch_product_data(api_url, params))

        scraped_products = scrape_product_data("https://best.aliexpress.com/search?q=laptop")
        all_products.extend(scraped_products)

        compare_prices(all_products)
        return jsonify(all_products)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
