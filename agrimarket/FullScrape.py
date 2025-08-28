import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
from urllib.parse import urlencode
from datetime import datetime, timedelta
import time
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
# Load commodity codes
commodities = pd.read_csv('agrimarket/CommodityAndCommodityHeads.csv')

def get_url(Commodity, CommodityHead, Market=0, date_from=None, date_to=None):
    base_url = 'https://agmarknet.gov.in/SearchCmmMkt.aspx'
    today = datetime.today()
    if date_to is None:
        date_to = today
    if date_from is None:
        date_from = today - timedelta(days=45)
    # If date_from is a string, use it directly; else, format it
    if isinstance(date_from, str):
        date_from_str = date_from
    else:
        date_from_str = date_from.strftime('%d-%b-%Y')
    if isinstance(date_to, str):
        date_to_str = date_to
    else:
        date_to_str = date_to.strftime('%d-%b-%Y')
    parameters = {
        "Tx_Commodity": Commodity,
        "Tx_State": "0",
        "Tx_District": 0,
        "Tx_Market": 0,  # All markets
        "DateFrom": date_from_str,
        "DateTo": date_to_str,
        "Fr_Date": date_from_str,
        "To_Date": date_to_str,
        "Tx_Trend": 0,
        "Tx_CommodityHead": CommodityHead,
        "Tx_StateHead": "--Select--",
        "Tx_DistrictHead": "--Select--",
        "Tx_MarketHead": "--Select--",
    }
    query = urlencode(parameters)
    return f"{base_url}?{query}"

def get_soup_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return bs(response.text, "html.parser")

def get_all_tables(soup):
    return soup.find_all("table")

def get_table_headers(table):
    headers = []
    for th in table.find("tr").find_all("th"):
        headers.append(th.text.strip())
    return headers

def get_table_rows(table):
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = []
        tds = tr.find_all("td")
        for td in tds:
            cells.append(td.text.strip())
        rows.append(cells)
    return rows

def fetch_commodity_data(commodity_code, commodity_head, date_from, date_to):
    url = get_url(commodity_code, commodity_head, Market=0, date_from=date_from, date_to=date_to)
    try:
        soup = get_soup_from_url(url)
        tables = get_all_tables(soup)
        if not tables:
            return []
        table = tables[0]
        headers = get_table_headers(table)
        rows = get_table_rows(table)
        return [dict(zip(headers, row)) for row in rows if len(row) == len(headers)]
    except Exception as e:
        print(f"Error fetching all markets - {commodity_code}: {e}")
        return []

def scrape_all():
    all_data = []
    today = datetime.today()
    from_date = today - timedelta(days=45)
    date_from_str = from_date.strftime('%d %b %Y')
    date_to_str = today.strftime('%d %b %Y')

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                fetch_commodity_data,
                commodity['Commodity'],
                commodity['CommodityHead'],
                date_from_str,
                date_to_str
            )
            for _, commodity in commodities.iterrows()
        ]
        for future in as_completed(futures):
            all_data.extend(future.result())

    if all_data:
        df = pd.DataFrame(all_data)
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            print("DATABASE_URL not set in .env file.")
            return
        try:
            engine = create_engine(db_url)
            df.to_sql("market_prices", engine, if_exists="append", index=False)
            print(f"Inserted {len(df)} rows into the market_prices table in the database")
            # --- Add indexes for fast queries ---
            with engine.connect() as conn:
                conn.execute(
                    text('CREATE INDEX IF NOT EXISTS idx_commodity ON market_prices("Commodity");')
                )
                conn.execute(
                    text('CREATE INDEX IF NOT EXISTS idx_district ON market_prices("District Name");')
                )
                conn.execute(
                    text('CREATE INDEX IF NOT EXISTS idx_commodity_district ON market_prices("Commodity", "District Name");')
                )
                print("Ensured indexes exist on Commodity and District Name columns.")
        except Exception as e:
            print(f"Error inserting data into database: {e}")
    else:
        print("No data scraped.")

if __name__ == "__main__":
    scrape_all()
