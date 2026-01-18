# ETL: World's Largest Banks Data 🏦

This project implements a complete ETL pipeline in Python to track the largest banks worldwide by market capitalization.

## Project Architecture
- **Extraction**: Web scraping from Wikipedia using BeautifulSoup.
- **Transformation**: Currency conversion (USD to GBP, EUR, INR) using exchange rate data.
- **Loading**: Storage in a local CSV file and an SQLite database.

## How to use
1. Clone the repo: `git clone ...`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python banks_project.py`

## Technologies Used
- Python 3.x
- Pandas & Numpy
- BeautifulSoup4
- SQLite3
