import os
import hashlib
import requests
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

def get_available_dates(base_url, coin_pair):
    """Fetch all available files for the given coin pair using Selenium."""
    url = urljoin(base_url, f"?prefix=data/spot/monthly/trades/{coin_pair}/")

    # Initialize Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run browser in headless mode
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    try:
        driver.get(url)

        # Wait until sufficient links are loaded
        timeout = 30  # Adjust as needed
        min_links = 10  # Minimum number of links expected
        WebDriverWait(driver, timeout).until(
            lambda d: len(d.find_elements(By.TAG_NAME, "a")) >= min_links
        )

        # Ensure lazy-loaded content is visible (optional: scroll)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Wait for any additional lazy-loaded content

        # Get page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract links to .zip files
        links = soup.find_all('a', href=True)
        files = [link['href'].split("/")[-1] for link in links if link['href'].endswith('.zip')]

        if not files:
            raise ValueError(f"No files found for {coin_pair} at {url}. Check the coin pair name or the base URL.")

        return files

    finally:
        driver.quit()

def download_file(url, dest_path):
    """Download a file from a URL to a local destination."""
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(dest_path)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))

def download_checksum(checksum_url, dest_path):
    """Download the checksum file from a URL to a local destination."""
    with requests.get(checksum_url, stream=True) as response:
        response.raise_for_status()
        with open(dest_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

def validate_checksum(file_path, checksum_file_path):
    """Validate the checksum of the downloaded file."""
    with open(checksum_file_path, 'r') as checksum_file:
        checksum = checksum_file.read().strip().split()[0]  # Extract checksum
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == checksum

def main():
    base_url = "https://data.binance.vision/"
    coin_pair = input("Enter the coin pair (e.g., BTCUSDT): ").upper()
    output_dir = f"data/{coin_pair}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        available_dates = get_available_dates(base_url, coin_pair)
        print(f"Found {len(available_dates)} files for {coin_pair}.")

        for date_file in available_dates:
            file_url = urljoin(base_url, f"data/spot/monthly/trades/{coin_pair}/{date_file}")
            checksum_url = file_url + ".CHECKSUM"
            local_file_path = os.path.join(output_dir, date_file)
            checksum_file_path = local_file_path + ".CHECKSUM"

            print(f"Downloading {date_file}...")
            download_file(file_url, local_file_path)
            
            print(f"Downloading checksum for {date_file}...")
            download_checksum(checksum_url, checksum_file_path)

            print(f"Validating checksum for {date_file}...")
            if validate_checksum(local_file_path, checksum_file_path):
                print(f"Checksum validated for {date_file}.")
            else:
                print(f"Checksum mismatch for {date_file}. Deleting file.")
                os.remove(local_file_path)
                os.remove(checksum_file_path)

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
