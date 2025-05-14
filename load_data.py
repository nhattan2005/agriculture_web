import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_to_db(csv_file):
    """
    Load data from CSV file to PostgreSQL database
    """
    try:
        # Read CSV file
        logging.info(f"Reading data from {csv_file}")
        df = pd.read_csv(csv_file)
        logging.info(f"Successfully loaded {len(df)} records from CSV")
        
        # Get database connection from environment variable
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            logging.error("DATABASE_URL environment variable not found")
            return False
        
        # Create SQLAlchemy engine
        engine = create_engine(db_url)
        
        # Insert data into database
        logging.info("Inserting data into database...")
        df.to_sql('crop_data', engine, if_exists='append', index=False)
        
        logging.info("Data has been successfully loaded into the database")
        return True
    
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return False

if __name__ == "__main__":
    # Path to CSV file
    csv_file = 'data/Crop_recommendation.csv'
    
    # Load data to database
    success = load_data_to_db(csv_file)
    
    if success:
        print("Data loaded successfully!")
    else:
        print("Failed to load data. Check the logs for details.")