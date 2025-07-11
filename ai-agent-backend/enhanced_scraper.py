import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

class EnhancedScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_driver(self, headless=True):
        """Initialize Chrome driver with optimal settings"""
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    
    def rate_limit(self, min_delay=1, max_delay=3):
        """Add random delay between requests"""
        time.sleep(random.uniform(min_delay, max_delay))
    
    def scrape_amazon(self, query, max_results=10):
        """Enhanced Amazon scraping with pagination"""
        products = []
        driver = self.get_driver()
        
        try:
            search_url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}"
            driver.get(search_url)
            
            # Wait for products to load
            WebDriverWait(driver, 10).wait(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-component-type="s-search-result"]'))
            )
            
            product_elements = driver.find_elements(By.CSS_SELECTOR, '[data-component-type="s-search-result"]')
            
            for i, element in enumerate(product_elements[:max_results]):
                try:
                    name_elem = element.find_element(By.CSS_SELECTOR, 'h2 a span')
                    price_elem = element.find_element(By.CSS_SELECTOR, '.a-price-whole')
                    rating_elem = element.find_element(By.CSS_SELECTOR, '.a-icon-alt')
                    image_elem = element.find_element(By.CSS_SELECTOR, 'img')
                    link_elem = element.find_element(By.CSS_SELECTOR, 'h2 a')
                    
                    product = {
                        'name': name_elem.text.strip(),
                        'price': float(price_elem.text.replace(',', '')),
                        'rating': float(rating_elem.get_attribute('innerHTML').split()[0]) if rating_elem else 0,
                        'image': image_elem.get_attribute('src'),
                        'url': 'https://amazon.com' + link_elem.get_attribute('href'),
                        'platform': 'Amazon'
                    }
                    products.append(product)
                    
                except Exception as e:
                    print(f"Error parsing Amazon product {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error scraping Amazon: {e}")
        finally:
            driver.quit()
            
        self.rate_limit()
        return products
    
    def scrape_ebay(self, query, max_results=10):
        """Enhanced eBay scraping"""
        products = []
        driver = self.get_driver()
        
        try:
            search_url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}"
            driver.get(search_url)
            
            product_elements = driver.find_elements(By.CSS_SELECTOR, '.s-item')[:max_results]
            
            for element in product_elements:
                try:
                    name = element.find_element(By.CSS_SELECTOR, '.s-item__title').text
                    price_text = element.find_element(By.CSS_SELECTOR, '.s-item__price').text
                    link = element.find_element(By.CSS_SELECTOR, '.s-item__link').get_attribute('href')
                    
                    # Extract price
                    price_cleaned = price_text.replace('$', '').replace(',', '').split()[0]
                    price = float(price_cleaned) if price_cleaned.replace('.', '').isdigit() else 0
                    
                    if price > 0 and 'Shop on eBay' not in name:
                        product = {
                            'name': name,
                            'price': price,
                            'url': link,
                            'platform': 'eBay',
                            'rating': 0  # eBay doesn't show ratings in search
                        }
                        products.append(product)
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error scraping eBay: {e}")
        finally:
            driver.quit()
            
        self.rate_limit()
        return products
    
    def scrape_walmart(self, query, max_results=10):
        """Enhanced Walmart scraping"""
        products = []
        driver = self.get_driver()
        
        try:
            search_url = f"https://www.walmart.com/search?q={query.replace(' ', '+')}"
            driver.get(search_url)
            
            # Wait for products to load
            time.sleep(3)
            
            product_elements = driver.find_elements(By.CSS_SELECTOR, '[data-testid="item"]')[:max_results]
            
            for element in product_elements:
                try:
                    name = element.find_element(By.CSS_SELECTOR, '[data-testid="product-title"]').text
                    price_elem = element.find_element(By.CSS_SELECTOR, '[itemprop="price"]')
                    price = float(price_elem.get_attribute('content'))
                    
                    link_elem = element.find_element(By.CSS_SELECTOR, 'a')
                    link = 'https://walmart.com' + link_elem.get_attribute('href')
                    
                    product = {
                        'name': name,
                        'price': price,
                        'url': link,
                        'platform': 'Walmart',
                        'rating': 0
                    }
                    products.append(product)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error scraping Walmart: {e}")
        finally:
            driver.quit()
            
        self.rate_limit()
        return products
    
    def scrape_all_platforms(self, query, max_results_per_platform=10):
        """Scrape multiple platforms concurrently"""
        all_products = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.scrape_amazon, query, max_results_per_platform): 'Amazon',
                executor.submit(self.scrape_ebay, query, max_results_per_platform): 'eBay',
                executor.submit(self.scrape_walmart, query, max_results_per_platform): 'Walmart'
            }
            
            for future in as_completed(futures):
                platform = futures[future]
                try:
                    products = future.result()
                    all_products.extend(products)
                    print(f"Scraped {len(products)} products from {platform}")
                except Exception as e:
                    print(f"Error scraping {platform}: {e}")
        
        return all_products
