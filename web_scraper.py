import requests
from bs4 import BeautifulSoup
import json
import re
import hashlib
import random
import os
import time
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed


class WebsiteScraper:
    def __init__(self, output_file="website_dataset.json", max_sites=1000):
        self.output_file = output_file
        self.max_sites = max_sites
        self.dataset = []
        self.visited_domains = set()

        # List of popular website categories and example sites
        self.website_categories = {
            "E-commerce": ["amazon", "ebay", "etsy", "walmart", "bestbuy", "aliexpress", "target"],
            "Social Media": ["twitter", "instagram", "facebook", "pinterest", "linkedin", "reddit", "tumblr"],
            "Technology": ["techcrunch", "wired", "cnet", "theverge", "engadget", "mashable", "gizmodo"],
            "News": ["cnn", "bbc", "nytimes", "guardian", "reuters", "washingtonpost", "bloomberg"],
            "Entertainment": ["netflix", "spotify", "hulu", "disneyplus", "youtube", "twitch", "imdb"],
            "Education": ["coursera", "udemy", "khanacademy", "edx", "duolingo", "chegg", "quizlet"],
            "Travel": ["booking", "airbnb", "expedia", "tripadvisor", "kayak", "hotels", "lonely-planet"],
            "Finance": ["paypal", "robinhood", "chase", "wellsfargo", "coinbase", "mint", "bankofamerica"],
            "Health": ["webmd", "mayoclinic", "healthline", "fitbit", "myfitnesspal", "everydayhealth", "medlineplus"],
            "Food": ["allrecipes", "epicurious", "foodnetwork", "yelp", "ubereats", "doordash", "grubhub"]
        }

        # Headers to mimic a browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

    def generate_object_id(self, url):
        """Generate a MongoDB-like ObjectId based on the URL"""
        return hashlib.md5(url.encode()).hexdigest()[:24]

    def extract_domain(self, url):
        """Extract the domain from a URL"""
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain

    def get_site_type(self, domain):
        """Determine site type based on domain or return random categories"""
        for category, sites in self.website_categories.items():
            for site in sites:
                if site in domain:
                    secondary_categories = list(self.website_categories.keys())
                    secondary_categories.remove(category)
                    secondary = random.choice(secondary_categories)
                    return f"{category.lower()}, {secondary.lower()}"

        # If not found, assign random categories
        categories = random.sample(list(self.website_categories.keys()), 2)
        return f"{categories[0].lower()}, {categories[1].lower()}"

    def extract_css(self, soup, base_url):
        """Extract CSS code from the webpage"""
        css_code = []

        # Extract inline CSS
        style_tags = soup.find_all('style')
        for style in style_tags:
            if style.string:
                css_code.append(style.string)

        # Extract external CSS
        css_links = soup.find_all('link', rel='stylesheet')
        for link in css_links:
            if 'href' in link.attrs:
                css_url = urljoin(base_url, link['href'])
                try:
                    response = requests.get(css_url, headers=self.headers, timeout=10)
                    if response.status_code == 200:
                        css_code.append(response.text)
                except Exception as e:
                    print(f"Error fetching CSS from {css_url}: {e}")

        return "\n".join(css_code)

    def extract_js(self, soup, base_url):
        """Extract JavaScript code from the webpage"""
        js_code = []

        # Extract inline JavaScript
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string:
                js_code.append(script.string)

        # Extract external JavaScript
        for script in script_tags:
            if 'src' in script.attrs:
                js_url = urljoin(base_url, script['src'])
                try:
                    response = requests.get(js_url, headers=self.headers, timeout=10)
                    if response.status_code == 200:
                        js_code.append(response.text)
                except Exception as e:
                    print(f"Error fetching JS from {js_url}: {e}")

        return "\n".join(js_code)

    def extract_images(self, soup, base_url):
        """Extract image URLs from the webpage"""
        image_urls = []

        # Find all img tags
        img_tags = soup.find_all('img')
        for img in img_tags:
            if 'src' in img.attrs:
                img_url = urljoin(base_url, img['src'])
                image_urls.append(img_url)

        # Find image links in CSS background
        style_tags = soup.find_all(['style', 'link'])
        css_text = ' '.join([str(tag) for tag in style_tags])
        url_pattern = r'url\([\'"]?(.*?)[\'"]?\)'
        for url in re.findall(url_pattern, css_text):
            if url and not url.startswith('data:'):
                img_url = urljoin(base_url, url)
                image_urls.append(img_url)

        # Limit to 20 unique images
        unique_images = list(set(image_urls))[:20]
        return unique_images

    def scrape_website(self, url):
        """Scrape a single website and return its data"""
        try:
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                print(f"Failed to fetch {url}: Status code {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            domain = self.extract_domain(url)

            css_code = self.extract_css(soup, url)
            js_code = self.extract_js(soup, url)
            images = self.extract_images(soup, url)

            # Create document
            document = {
                "Object_Id": self.generate_object_id(url),
                "Name": domain,
                "Type": self.get_site_type(domain),
                "css-code": css_code,
                "js-code": js_code,
                "Images_and_icons": images
            }

            return document

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def generate_sites_to_scrape(self):
        """Generate a list of websites to scrape"""
        sites_to_scrape = []

        # Add known sites from categories
        for category, sites in self.website_categories.items():
            for site in sites:
                sites_to_scrape.append(f"www.{site}.com")

        # Add the top Alexa sites (these are examples and may need updating)
        top_sites = [
            "google.com", "youtube.com", "baidu.com", "wikipedia.org", "yahoo.com",
            "qq.com", "taobao.com", "tmall.com", "sohu.com", "facebook.com",
            "jd.com", "360.cn", "amazon.com", "weibo.com", "sina.com.cn",
            "pages.tmall.com", "live.com", "netflix.com", "reddit.com", "vk.com"
        ]
        sites_to_scrape.extend(top_sites)

        # Remove duplicates
        return list(set(sites_to_scrape))

    def save_dataset(self):
        """Save the dataset to a JSON file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)

        print(f"Dataset saved to {self.output_file} with {len(self.dataset)} websites")

    def scrape_websites(self):
        """Scrape multiple websites in parallel"""
        sites_to_scrape = self.generate_sites_to_scrape()
        random.shuffle(sites_to_scrape)  # Randomize the order

        print(f"Starting to scrape {len(sites_to_scrape)} websites...")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for site in sites_to_scrape[:self.max_sites]:
                futures.append(executor.submit(self.scrape_website, site))

            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.dataset.append(result)
                    print(f"Scraped: {result['Name']} - Total: {len(self.dataset)}/{self.max_sites}")

                    # Save intermediately every 50 sites
                    if len(self.dataset) % 50 == 0:
                        self.save_dataset()

                    if len(self.dataset) >= self.max_sites:
                        break

        # Final save
        self.save_dataset()

    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic data to reach desired dataset size"""
        print(f"Generating {num_samples} synthetic website samples...")

        # Template CSS styles
        css_templates = [
            """
            body { font-family: 'Arial', sans-serif; margin: 0; padding: 0; }
            .container { max-width: 1200px; margin: 0 auto; padding: 0 15px; }
            header { background-color: #333; color: white; padding: 20px 0; }
            .nav { display: flex; justify-content: space-between; align-items: center; }
            .logo { font-size: 24px; font-weight: bold; }
            .menu { display: flex; list-style: none; }
            .menu li { margin-left: 20px; }
            .menu a { color: white; text-decoration: none; }
            .hero { background-color: #f5f5f5; padding: 60px 0; text-align: center; }
            .hero h1 { font-size: 48px; margin-bottom: 20px; }
            .button { display: inline-block; background-color: #007bff; color: white; padding: 12px 24px; border-radius: 4px; text-decoration: none; }
            """,
            """
            :root { --primary: #ff6b6b; --secondary: #4ecdc4; --dark: #1a535c; --light: #f7fff7; }
            body { font-family: 'Montserrat', sans-serif; line-height: 1.6; color: var(--dark); }
            .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; }
            .card { border-radius: 8px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.3s; }
            .card:hover { transform: translateY(-5px); }
            .card img { width: 100%; height: auto; }
            .card-content { padding: 20px; }
            .btn { background: var(--primary); color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            """
        ]

        # Template JS snippets
        js_templates = [
            """
            document.addEventListener('DOMContentLoaded', function() {
                const toggleBtn = document.querySelector('.toggle-menu');
                const mobileMenu = document.querySelector('.mobile-menu');

                toggleBtn.addEventListener('click', function() {
                    mobileMenu.classList.toggle('active');
                });

                // Smooth scrolling
                document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                    anchor.addEventListener('click', function (e) {
                        e.preventDefault();
                        document.querySelector(this.getAttribute('href')).scrollIntoView({
                            behavior: 'smooth'
                        });
                    });
                });
            });
            """,
            """
            // Carousel functionality
            class Carousel {
                constructor(element) {
                    this.element = element;
                    this.slides = element.querySelectorAll('.slide');
                    this.currentIndex = 0;
                    this.init();
                }

                init() {
                    setInterval(() => this.nextSlide(), 5000);
                    this.showSlide(this.currentIndex);
                }

                showSlide(index) {
                    this.slides.forEach(slide => slide.classList.remove('active'));
                    this.slides[index].classList.add('active');
                }

                nextSlide() {
                    this.currentIndex = (this.currentIndex + 1) % this.slides.length;
                    this.showSlide(this.currentIndex);
                }
            }

            document.addEventListener('DOMContentLoaded', function() {
                const carousels = document.querySelectorAll('.carousel');
                carousels.forEach(carousel => new Carousel(carousel));
            });
            """
        ]

        # Random site name generator components
        prefixes = ["tech", "digital", "web", "net", "cyber", "data", "info", "meta", "smart", "cloud", "pro", "global",
                    "hyper", "next", "future"]
        roots = ["connect", "logic", "system", "work", "ware", "tech", "app", "hub", "spot", "base", "zone", "grid",
                 "flow", "pulse", "space"]
        suffixes = ["hub", "pro", "io", "tech", "labs", "space", "now", "app", "net", "web", "cloud", "ware", "core",
                    "base", "link"]
        tlds = [".com", ".io", ".net", ".org", ".app", ".co", ".tech", ".digital", ".ai", ".dev"]

        # Image URL templates
        image_url_templates = [
            "https://example.com/images/logo.png",
            "https://placeholder.com/300x200/",
            "https://picsum.photos/id/{}/200/300",
            "https://source.unsplash.com/random/300x200/?{}"
        ]

        # Categories to choose from
        all_categories = list(self.website_categories.keys())

        for i in range(num_samples):
            # Generate random site name
            prefix = random.choice(prefixes)
            root = random.choice(roots)
            suffix = random.choice(suffixes)
            tld = random.choice(tlds)

            # Combine name components with different patterns
            name_pattern = random.randint(1, 4)
            if name_pattern == 1:
                site_name = f"{prefix}{root}{tld}"
            elif name_pattern == 2:
                site_name = f"{prefix}-{root}{tld}"
            elif name_pattern == 3:
                site_name = f"{prefix}{suffix}{tld}"
            else:
                site_name = f"{root}{suffix}{tld}"

            # Remove domain extensions for name field
            display_name = site_name.replace(tld, "")

            # Generate random types (2 random categories)
            categories = random.sample(all_categories, 2)
            site_type = f"{categories[0].lower()}, {categories[1].lower()}"

            # Select random CSS and JS templates and add variations
            css = random.choice(css_templates)
            js = random.choice(js_templates)

            # Add color variations
            colors = [
                "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for _ in range(3)
            ]
            for color in colors:
                css = css.replace("#333", color, 1)

            # Generate random image URLs
            images = []
            for _ in range(random.randint(5, 15)):
                template = random.choice(image_url_templates)
                if "{}" in template:
                    if "picsum" in template:
                        template = template.format(random.randint(1, 1000))
                    elif "unsplash" in template:
                        topics = ["nature", "technology", "business", "food", "travel", "fashion"]
                        template = template.format(random.choice(topics))
                images.append(template)

            # Create synthetic document
            document = {
                "Object_Id": self.generate_object_id(site_name),
                "Name": display_name,
                "Type": site_type,
                "css-code": css,
                "js-code": js,
                "Images_and_icons": images
            }

            self.dataset.append(document)

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} synthetic samples")

        # Save the final dataset
        self.save_dataset()


def main():
    # Create and run the scraper
    scraper = WebsiteScraper(output_file="website_dataset.json", max_sites=1000)

    # Try to scrape real websites first
    scraper.scrape_websites()

    # If we didn't get enough sites, generate synthetic data to reach 1000
    if len(scraper.dataset) < 1000:
        remaining = 1000 - len(scraper.dataset)
        scraper.generate_synthetic_data(num_samples=remaining)


if __name__ == "__main__":
    main()