import json
import sqlite3
from datetime import datetime
from websocket import WebSocketApp
import threading

# SQLite database setup
db_name = "crypto_data.db"

# Create table and indexes outside of threads
def initialize_database():
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS coins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        coin_name TEXT NOT NULL,
        open_price REAL,
        high_price REAL,
        low_price REAL,
        close_price REAL,
        volume REAL,
        start_time TEXT,
        end_time TEXT,
        sma REAL
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_coin_name ON coins (coin_name);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON coins (start_time);")
    conn.commit()
    conn.close()

# Function to handle incoming WebSocket messages
def on_message(ws, message, db_name):
    data = json.loads(message)
    if 'k' in data:  # Kline data
        kline = data['k']
        open_price = float(kline['o'])
        high_price = float(kline['h'])
        low_price = float(kline['l'])
        close_price = float(kline['c'])
        volume = float(kline['v'])
        start_time = datetime.fromtimestamp(kline['t'] / 1000.0).isoformat()
        end_time = datetime.fromtimestamp(kline['T'] / 1000.0).isoformat()
        coin_name = data['s']
        # Calculate a simple moving average (example)
        sma = (open_price + close_price) / 2

        # Insert data into SQLite database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO coins 
            (coin_name, open_price, high_price, low_price, close_price, volume, start_time, end_time, sma) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (coin_name, open_price, high_price, low_price, close_price, volume, start_time, end_time, sma)
        )
        conn.commit()
        conn.close()
        print(f"Inserted data: {coin_name}, Start={start_time}, End={end_time}, SMA={sma}, Close={close_price}")

# Function to handle WebSocket errors
def on_error(ws, error):
    print(f"Error: {error}")

# Function to handle WebSocket closure
def on_close(ws, close_status_code, close_msg):
    print(f"Connection closed with code {close_status_code}: {close_msg}")

# Function to handle WebSocket connection
def on_open(ws):
    print("WebSocket connection opened")

# Function to start WebSocket for a specific URL
def start_stream(url, db_name):
    ws = WebSocketApp(
        url,
        on_message=lambda ws, msg: on_message(ws, msg, db_name),
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever()

# Main script
if __name__ == "__main__":
    # Initialize database
    initialize_database()

    # List of WebSocket streams for different coins
    streams = [
        "wss://stream.binance.com:9443/ws/ethusdt@kline_1s",
        "wss://stream.binance.com:9443/ws/btcusdt@kline_1s",
        "wss://stream.binance.com:9443/ws/solusdt@kline_1s",
        "wss://stream.binance.com:9443/ws/adausdt@kline_1s",
        "wss://stream.binance.com:9443/ws/dogeusdt@kline_1s"
    ]

    # Start a thread for each WebSocket connection
    threads = []
    for stream in streams:
        thread = threading.Thread(target=start_stream, args=(stream, db_name))
        thread.start()
        threads.append(thread)

    # Join threads to keep the main program running
    for thread in threads:
        thread.join()
