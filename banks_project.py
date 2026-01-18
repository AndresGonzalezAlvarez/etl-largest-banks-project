"""
ETL Pipeline for World's Largest Banks Data
============================================

This module implements a complete ETL (Extract, Transform, Load) pipeline
to extract bank data from Wikipedia, transform it with currency conversions,
and load it into both CSV files and SQLite database.

Author: ETL Project
Date: 2026
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime
from tabulate import tabulate
import warnings
import logging
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager
import os
import sys

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Suppress urllib3 OpenSSL warning
warnings.filterwarnings('ignore', message='.*urllib3.*')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for the ETL pipeline."""
    # URLs
    URL = 'https://web.archive.org/web/20230908091635/https://en.wikipedia.org/wiki/List_of_largest_banks'
    CSV_EXCHANGE = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-PY0221EN-Coursera/labs/v2/exchange_rate.csv'
    
    # Data attributes
    TABLE_ATTRIBUTES = ["Name", "MC_USD_Billion"]
    
    # File paths
    OUTPUT_CSV_PATH = './Largest_banks_data.csv'
    DATABASE_NAME = 'Banks.db'
    TABLE_NAME = 'Largest_banks'
    LOG_FILE = 'code_log.txt'
    
    # Request settings
    REQUEST_TIMEOUT = 30
    REQUEST_RETRIES = 3
    
    # Validation settings
    MIN_BANKS_EXPECTED = 5
    MIN_MARKET_CAP = 0.1  # Minimum market cap in billions USD

# ============================================================================
# LOGGING SETUP
# ============================================================================

class ETLLogger:
    """Enhanced logging system for ETL pipeline."""
    
    def __init__(self, log_file: str = Config.LOG_FILE):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging to both file and console."""
        # Create logger
        self.logger = logging.getLogger('ETL_Pipeline')
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s', 
                                       datefmt='%Y-%b-%d-%H:%M:%S')
        file_handler.setFormatter(file_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_format)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log(self, message: str, level: str = 'INFO'):
        """Log a message with timestamp."""
        if level.upper() == 'INFO':
            self.logger.info(message)
        elif level.upper() == 'WARNING':
            self.logger.warning(message)
        elif level.upper() == 'ERROR':
            self.logger.error(message)
        elif level.upper() == 'DEBUG':
            self.logger.debug(message)
    
    def log_progress(self, message: str):
        """Log progress message (backward compatibility)."""
        self.log(message, 'INFO')

# Initialize global logger
logger = ETLLogger()

# ============================================================================
# ETL PIPELINE CLASSES
# ============================================================================

class DataExtractor:
    """Handles data extraction from web sources."""
    
    def __init__(self, url: str, timeout: int = Config.REQUEST_TIMEOUT):
        self.url = url
        self.timeout = timeout
    
    def fetch_page(self) -> Optional[str]:
        """Fetch HTML content from URL with retry logic."""
        for attempt in range(Config.REQUEST_RETRIES):
            try:
                logger.log(f"Fetching data from URL (attempt {attempt + 1}/{Config.REQUEST_RETRIES})...")
                response = requests.get(self.url, timeout=self.timeout)
                response.raise_for_status()
                logger.log(f"Successfully fetched page ({len(response.text)} characters)")
                return response.text
            except requests.exceptions.RequestException as e:
                logger.log(f"Request failed (attempt {attempt + 1}): {str(e)}", 'WARNING')
                if attempt == Config.REQUEST_RETRIES - 1:
                    logger.log(f"Failed to fetch page after {Config.REQUEST_RETRIES} attempts", 'ERROR')
                    return None
        return None
    
    def extract_banks_data(self, table_attribs: List[str]) -> Optional[pd.DataFrame]:
        """
        Extract bank names and market cap from Wikipedia table.
        
        Args:
            table_attribs: List of column names for the DataFrame
            
        Returns:
            DataFrame with bank data or None if extraction fails
        """
        logger.log("=" * 60)
        logger.log("PHASE 1: EXTRACTION")
        logger.log("=" * 60)
        logger.log("Starting extraction process...")
        
        try:
            # Fetch page
            page_content = self.fetch_page()
            if not page_content:
                return None
            
            # Parse HTML
            logger.log("Parsing HTML content...")
            data = BeautifulSoup(page_content, 'html.parser')
            
            # Find tables
            tables = data.find_all('tbody')
            if not tables:
                logger.log("ERROR: No tables found on the page.", 'ERROR')
                return None
            
            logger.log(f"Found {len(tables)} table(s), processing first table...")
            
            # Initialize DataFrame
            df = pd.DataFrame(columns=table_attribs)
            rows = tables[0].find_all('tr')
            
            extracted_count = 0
            skipped_count = 0
            
            # Process each row
            for idx, row in enumerate(rows, 1):
                col = row.find_all('td')
                if len(col) >= 3:
                    try:
                        # Extract bank name
                        bank_name = col[1].get_text(strip=True)
                        
                        # Extract and clean market cap
                        market_cap_str = col[2].get_text(strip=True)
                        market_cap_str = market_cap_str.replace(',', '').replace('$', '').strip()
                        
                        # Validate and convert
                        if not bank_name or not market_cap_str:
                            skipped_count += 1
                            continue
                        
                        market_cap = float(market_cap_str)
                        
                        # Validate market cap is reasonable
                        if market_cap < Config.MIN_MARKET_CAP:
                            logger.log(f"Skipping row {idx}: Market cap too low ({market_cap})", 'DEBUG')
                            skipped_count += 1
                            continue
                        
                        # Add to DataFrame
                        new_row = pd.DataFrame({
                            "Name": [bank_name], 
                            "MC_USD_Billion": [market_cap]
                        })
                        df = pd.concat([df, new_row], ignore_index=True)
                        extracted_count += 1
                        
                    except (ValueError, IndexError, AttributeError) as e:
                        skipped_count += 1
                        logger.log(f"Skipping row {idx}: {str(e)}", 'DEBUG')
                        continue
            
            # Validate extraction results
            if df.empty:
                logger.log("ERROR: No valid data extracted from the table.", 'ERROR')
                return None
            
            if len(df) < Config.MIN_BANKS_EXPECTED:
                logger.log(f"WARNING: Only {len(df)} banks extracted (expected at least {Config.MIN_BANKS_EXPECTED})", 'WARNING')
            
            logger.log(f"✓ Extraction successful: {extracted_count} banks extracted, {skipped_count} rows skipped")
            logger.log(f"  Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
            logger.log(f"  Market cap range: ${df['MC_USD_Billion'].min():.2f}B - ${df['MC_USD_Billion'].max():.2f}B")
            
            return df
            
        except Exception as e:
            logger.log(f"ERROR: Extraction failed: {str(e)}", 'ERROR')
            import traceback
            logger.log(traceback.format_exc(), 'DEBUG')
            return None


class DataTransformer:
    """Handles data transformation including currency conversion."""
    
    def __init__(self, exchange_csv_url: str):
        self.exchange_csv_url = exchange_csv_url
        self.rates: Optional[Dict[str, float]] = None
    
    def load_exchange_rates(self) -> bool:
        """Load exchange rates from CSV file."""
        try:
            logger.log(f"Loading exchange rates from: {self.exchange_csv_url}")
            exchange_rate = pd.read_csv(self.exchange_csv_url)
            
            # Validate required currencies
            required_currencies = ['GBP', 'EUR', 'INR']
            available_currencies = exchange_rate['Currency'].tolist()
            
            missing = [c for c in required_currencies if c not in available_currencies]
            if missing:
                logger.log(f"ERROR: Missing exchange rates for: {missing}", 'ERROR')
                return False
            
            self.rates = exchange_rate.set_index('Currency').to_dict()['Rate']
            logger.log(f"✓ Exchange rates loaded successfully:")
            for currency, rate in self.rates.items():
                logger.log(f"  {currency}: {rate:.4f}")
            
            return True
            
        except Exception as e:
            logger.log(f"ERROR: Failed to load exchange rates: {str(e)}", 'ERROR')
            return False
    
    def transform_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Transform data by adding currency conversions.
        
        Args:
            df: DataFrame with bank data in USD
            
        Returns:
            Transformed DataFrame with multiple currencies or None if transformation fails
        """
        logger.log("=" * 60)
        logger.log("PHASE 2: TRANSFORMATION")
        logger.log("=" * 60)
        logger.log("Starting transformation process...")
        
        try:
            # Load exchange rates if not already loaded
            if self.rates is None:
                if not self.load_exchange_rates():
                    return None
            
            # Validate input data
            if df is None or df.empty:
                logger.log("ERROR: No data to transform", 'ERROR')
                return None
            
            if 'MC_USD_Billion' not in df.columns:
                logger.log("ERROR: Missing required column 'MC_USD_Billion'", 'ERROR')
                return None
            
            # Create a copy to avoid modifying original
            df_transformed = df.copy()
            
            # Calculate conversions
            logger.log("Calculating currency conversions...")
            df_transformed['MC_GBP_Billion'] = np.round(
                df_transformed['MC_USD_Billion'] * self.rates['GBP'], 2
            )
            df_transformed['MC_EUR_Billion'] = np.round(
                df_transformed['MC_USD_Billion'] * self.rates['EUR'], 2
            )
            df_transformed['MC_INR_Billion'] = np.round(
                df_transformed['MC_USD_Billion'] * self.rates['INR'], 2
            )
            
            # Validate results
            logger.log("✓ Transformation successful")
            logger.log(f"  Added columns: MC_GBP_Billion, MC_EUR_Billion, MC_INR_Billion")
            logger.log(f"  Final data shape: {df_transformed.shape[0]} rows × {df_transformed.shape[1]} columns")
            
            # Show sample statistics
            logger.log(f"  Sample conversion (first bank):")
            first_row = df_transformed.iloc[0]
            logger.log(f"    USD: ${first_row['MC_USD_Billion']:.2f}B")
            logger.log(f"    GBP: £{first_row['MC_GBP_Billion']:.2f}B")
            logger.log(f"    EUR: €{first_row['MC_EUR_Billion']:.2f}B")
            logger.log(f"    INR: ₹{first_row['MC_INR_Billion']:.2f}B")
            
            return df_transformed
            
        except Exception as e:
            logger.log(f"ERROR: Transformation failed: {str(e)}", 'ERROR')
            import traceback
            logger.log(traceback.format_exc(), 'DEBUG')
            return df  # Return original data if transformation fails


class DataLoader:
    """Handles loading data to CSV and database."""
    
    @staticmethod
    def load_to_csv(df: pd.DataFrame, output_path: str) -> bool:
        """
        Save DataFrame to CSV file with backup.
        
        Args:
            df: DataFrame to save
            output_path: Path to output CSV file
            
        Returns:
            True if successful, False otherwise
        """
        logger.log("=" * 60)
        logger.log("PHASE 3: LOADING TO CSV")
        logger.log("=" * 60)
        
        try:
            # Create backup if file exists
            if os.path.exists(output_path):
                backup_path = f"{output_path}.backup"
                logger.log(f"Creating backup: {backup_path}")
                os.rename(output_path, backup_path)
            
            logger.log(f"Saving data to CSV: {output_path}")
            df.to_csv(output_path, index=False)
            
            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.log(f"✓ CSV file saved successfully ({file_size:,} bytes)")
                return True
            else:
                logger.log("ERROR: CSV file was not created", 'ERROR')
                return False
                
        except Exception as e:
            logger.log(f"ERROR: Failed to save CSV: {str(e)}", 'ERROR')
            return False
    
    @staticmethod
    @contextmanager
    def database_connection(db_name: str):
        """
        Context manager for database connections.
        
        Args:
            db_name: Name of the database file
            
        Yields:
            SQLite connection object
        """
        conn = None
        try:
            logger.log(f"Establishing database connection: {db_name}")
            conn = sqlite3.connect(db_name)
            yield conn
            conn.commit()
            logger.log("Database transaction committed")
        except Exception as e:
            if conn:
                conn.rollback()
                logger.log(f"ERROR: Database transaction rolled back: {str(e)}", 'ERROR')
            raise
        finally:
            if conn:
                conn.close()
                logger.log("Database connection closed")
    
    @staticmethod
    def load_to_db(df: pd.DataFrame, sql_connection: sqlite3.Connection, 
                   table_name: str) -> bool:
        """
        Load DataFrame into SQLite database.
        
        Args:
            df: DataFrame to load
            sql_connection: SQLite connection object
            table_name: Name of the table
            
        Returns:
            True if successful, False otherwise
        """
        logger.log("=" * 60)
        logger.log("PHASE 4: LOADING TO DATABASE")
        logger.log("=" * 60)
        
        try:
            logger.log(f"Loading data to SQL table: {table_name}")
            
            # Get row count before insertion
            cursor = sql_connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            old_count = cursor.fetchone()[0] if cursor.fetchone() else 0
            
            # Load data
            df.to_sql(table_name, sql_connection, if_exists='replace', index=False)
            
            # Verify insertion
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            new_count = cursor.fetchone()[0]
            
            logger.log(f"✓ Data loaded successfully to database")
            logger.log(f"  Table: {table_name}")
            logger.log(f"  Rows inserted: {new_count}")
            logger.log(f"  Columns: {', '.join(df.columns.tolist())}")
            
            return True
            
        except Exception as e:
            logger.log(f"ERROR: Failed to load data to database: {str(e)}", 'ERROR')
            return False


class QueryExecutor:
    """Handles SQL query execution and result display."""
    
    def __init__(self, sql_connection: sqlite3.Connection):
        self.conn = sql_connection
    
    def execute_query(self, query: str, description: str = "") -> Optional[pd.DataFrame]:
        """
        Execute SQL query and return results.
        
        Args:
            query: SQL query string
            description: Optional description of the query
            
        Returns:
            DataFrame with query results or None if query fails
        """
        try:
            if description:
                logger.log(f"Executing query: {description}")
            query_output = pd.read_sql(query, self.conn)
            return query_output
        except Exception as e:
            logger.log(f"ERROR: Query failed: {str(e)}", 'ERROR')
            return None
    
    def display_query_results(self, query: str, description: str = ""):
        """
        Execute and display SQL query results in formatted table.
        
        Args:
            query: SQL query string
            description: Optional description of the query
        """
        print(f"\n{'=' * 80}")
        if description:
            print(f"[QUERY]: {description}")
        else:
            print(f"[QUERY]: {query}")
        print(f"{'=' * 80}")
        
        result = self.execute_query(query, description)
        if result is not None and not result.empty:
            print(tabulate(result, headers='keys', tablefmt='psql', showindex=False))
            print(f"\nRows returned: {len(result)}")
        else:
            print("No results returned.")


class DataVisualizer:
    """Creates beautiful visualizations of bank data."""
    
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = output_dir
        if not VISUALIZATION_AVAILABLE:
            logger.log("WARNING: matplotlib/seaborn not available. Visualizations disabled.", 'WARNING')
            return
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
    
    def create_all_visualizations(self, df: pd.DataFrame) -> bool:
        """
        Create all visualizations for the bank data.
        
        Args:
            df: DataFrame with bank data
            
        Returns:
            True if successful, False otherwise
        """
        if not VISUALIZATION_AVAILABLE:
            return False
        
        logger.log("=" * 60)
        logger.log("PHASE 6: DATA VISUALIZATION")
        logger.log("=" * 60)
        logger.log("Creating visualizations...")
        
        try:
            # Sort by USD market cap for better visualization
            df_sorted = df.sort_values('MC_USD_Billion', ascending=True)
            
            # Create all visualizations
            self._create_top_banks_bar_chart(df_sorted)
            self._create_currency_comparison_chart(df_sorted)
            self._create_market_cap_distribution(df_sorted)
            self._create_currency_heatmap(df_sorted)
            self._create_dashboard(df_sorted)
            
            logger.log(f"✓ All visualizations created successfully in '{self.output_dir}/'")
            return True
            
        except Exception as e:
            logger.log(f"ERROR: Visualization failed: {str(e)}", 'ERROR')
            import traceback
            logger.log(traceback.format_exc(), 'DEBUG')
            return False
    
    def _create_top_banks_bar_chart(self, df: pd.DataFrame):
        """Create horizontal bar chart of top banks by market cap (USD)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top 10 banks
        top_banks = df.tail(10)
        
        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_banks)))
        bars = ax.barh(top_banks['Name'], top_banks['MC_USD_Billion'], color=colors)
        
        # Customize
        ax.set_xlabel('Market Capitalization (USD Billions)', fontweight='bold')
        ax.set_title('Top 10 Largest Banks by Market Cap (USD)', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(top_banks.iterrows()):
            value = row['MC_USD_Billion']
            ax.text(value + 5, i, f'${value:.2f}B', va='center', fontweight='bold')
        
        # Invert y-axis to show largest at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_top_banks_usd.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.log("  ✓ Created: Top banks bar chart (USD)")
    
    def _create_currency_comparison_chart(self, df: pd.DataFrame):
        """Create grouped bar chart comparing all currencies."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get top 8 banks for readability
        top_banks = df.tail(8)
        x = np.arange(len(top_banks))
        width = 0.2
        
        # Normalize values for comparison (divide by 100 for better scale)
        usd_values = top_banks['MC_USD_Billion'].values / 100
        gbp_values = top_banks['MC_GBP_Billion'].values / 100
        eur_values = top_banks['MC_EUR_Billion'].values / 100
        
        # Create bars
        bars1 = ax.bar(x - width, usd_values, width, label='USD', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x, gbp_values, width, label='GBP', color='#A23B72', alpha=0.8)
        bars3 = ax.bar(x + width, eur_values, width, label='EUR', color='#F18F01', alpha=0.8)
        
        # Customize
        ax.set_xlabel('Bank', fontweight='bold')
        ax.set_ylabel('Market Capitalization (Billions / 100)', fontweight='bold')
        ax.set_title('Market Cap Comparison: USD vs GBP vs EUR', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in top_banks['Name']], 
                           rotation=45, ha='right')
        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_currency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.log("  ✓ Created: Currency comparison chart")
    
    def _create_market_cap_distribution(self, df: pd.DataFrame):
        """Create distribution/histogram of market caps."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(df['MC_USD_Billion'], bins=15, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Market Cap (USD Billions)', fontweight='bold')
        ax1.set_ylabel('Number of Banks', fontweight='bold')
        ax1.set_title('Distribution of Market Capitalization', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Box plot
        box_data = [df['MC_USD_Billion'], df['MC_GBP_Billion'], df['MC_EUR_Billion']]
        bp = ax2.boxplot(box_data, labels=['USD', 'GBP', 'EUR'], patch_artist=True)
        colors_box = ['#2E86AB', '#A23B72', '#F18F01']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Market Cap (Billions)', fontweight='bold')
        ax2.set_title('Market Cap by Currency (Box Plot)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_market_cap_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.log("  ✓ Created: Market cap distribution charts")
    
    def _create_currency_heatmap(self, df: pd.DataFrame):
        """Create heatmap showing correlation between currencies."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data for heatmap (top 10 banks)
        top_banks = df.tail(10)
        heatmap_data = top_banks[['MC_USD_Billion', 'MC_GBP_Billion', 'MC_EUR_Billion', 'MC_INR_Billion']].T
        heatmap_data.columns = [name[:20] + '...' if len(name) > 20 else name for name in top_banks['Name']]
        
        # Normalize data for better visualization (divide by max)
        heatmap_data_norm = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
        
        # Create heatmap
        sns.heatmap(heatmap_data_norm, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Normalized Market Cap'}, ax=ax, linewidths=0.5)
        
        ax.set_title('Market Cap Heatmap: Top 10 Banks Across Currencies', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Bank', fontweight='bold')
        ax.set_ylabel('Currency', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_currency_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.log("  ✓ Created: Currency heatmap")
    
    def _create_dashboard(self, df: pd.DataFrame):
        """Create a comprehensive dashboard with multiple visualizations."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Top 5 banks pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        top_5 = df.tail(5)
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(top_5)))
        ax1.pie(top_5['MC_USD_Billion'], labels=[name[:15] + '...' if len(name) > 15 else name 
                                                  for name in top_5['Name']], 
               autopct='%1.1f%%', startangle=90, colors=colors_pie)
        ax1.set_title('Top 5 Banks Market Share', fontweight='bold')
        
        # 2. Top 10 horizontal bars
        ax2 = fig.add_subplot(gs[0, 1:])
        top_10 = df.tail(10)
        ax2.barh(range(len(top_10)), top_10['MC_USD_Billion'], color=plt.cm.viridis(np.linspace(0, 1, len(top_10))))
        ax2.set_yticks(range(len(top_10)))
        ax2.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in top_10['Name']], fontsize=8)
        ax2.set_xlabel('Market Cap (USD Billions)', fontweight='bold')
        ax2.set_title('Top 10 Banks by Market Cap', fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Currency comparison (stacked area)
        ax3 = fig.add_subplot(gs[1, :])
        top_8 = df.tail(8)
        x_pos = range(len(top_8))
        ax3.fill_between(x_pos, 0, top_8['MC_USD_Billion'], label='USD', alpha=0.6, color='#2E86AB')
        ax3.fill_between(x_pos, top_8['MC_USD_Billion'], 
                        top_8['MC_USD_Billion'] + top_8['MC_GBP_Billion'], 
                        label='GBP', alpha=0.6, color='#A23B72')
        ax3.fill_between(x_pos, top_8['MC_USD_Billion'] + top_8['MC_GBP_Billion'],
                        top_8['MC_USD_Billion'] + top_8['MC_GBP_Billion'] + top_8['MC_EUR_Billion'],
                        label='EUR', alpha=0.6, color='#F18F01')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([name[:12] + '...' if len(name) > 12 else name for name in top_8['Name']], 
                           rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Market Cap (Billions)', fontweight='bold')
        ax3.set_title('Currency Comparison (Stacked)', fontweight='bold')
        ax3.legend(loc='upper left')
        ax3.grid(alpha=0.3)
        
        # 4. Statistics summary
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis('off')
        stats_text = f"""
        STATISTICS SUMMARY
        {'='*30}
        Total Banks: {len(df)}
        
        USD Market Cap:
        • Max: ${df['MC_USD_Billion'].max():.2f}B
        • Min: ${df['MC_USD_Billion'].min():.2f}B
        • Avg: ${df['MC_USD_Billion'].mean():.2f}B
        • Total: ${df['MC_USD_Billion'].sum():.2f}B
        
        GBP Market Cap:
        • Avg: £{df['MC_GBP_Billion'].mean():.2f}B
        
        EUR Market Cap:
        • Avg: €{df['MC_EUR_Billion'].mean():.2f}B
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=9, family='monospace', 
                verticalalignment='center', fontweight='bold')
        
        # 5. Scatter plot: USD vs EUR
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.scatter(df['MC_USD_Billion'], df['MC_EUR_Billion'], 
                   s=100, alpha=0.6, c=df['MC_USD_Billion'], cmap='viridis')
        ax5.set_xlabel('USD Market Cap (Billions)', fontweight='bold')
        ax5.set_ylabel('EUR Market Cap (Billions)', fontweight='bold')
        ax5.set_title('USD vs EUR Correlation', fontweight='bold')
        ax5.grid(alpha=0.3)
        
        # 6. Top 3 comparison
        ax6 = fig.add_subplot(gs[2, 2])
        top_3 = df.tail(3)
        currencies = ['USD', 'GBP', 'EUR']
        x = np.arange(len(currencies))
        width = 0.25
        
        for i, (idx, bank) in enumerate(top_3.iterrows()):
            values = [bank['MC_USD_Billion'], bank['MC_GBP_Billion'], bank['MC_EUR_Billion']]
            # Normalize for comparison
            values_norm = [v / max(values) for v in values]
            ax6.bar(x + i*width, values_norm, width, label=bank['Name'][:15], alpha=0.8)
        
        ax6.set_ylabel('Normalized Market Cap', fontweight='bold')
        ax6.set_title('Top 3 Banks Comparison', fontweight='bold')
        ax6.set_xticks(x + width)
        ax6.set_xticklabels(currencies)
        ax6.legend(fontsize=7)
        ax6.grid(axis='y', alpha=0.3)
        
        # Main title
        fig.suptitle('World\'s Largest Banks - Comprehensive Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(f'{self.output_dir}/05_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.log("  ✓ Created: Comprehensive dashboard")


# ============================================================================
# VERIFICATION QUERIES
# ============================================================================

VERIFICATION_QUERIES = [
    {
        "query": f"SELECT * FROM {Config.TABLE_NAME}",
        "description": "Display all bank data"
    },
    {
        "query": f"SELECT AVG(MC_GBP_Billion) AS Average_GBP FROM {Config.TABLE_NAME}",
        "description": "Calculate average market cap in GBP"
    },
    {
        "query": f"SELECT Name FROM {Config.TABLE_NAME} LIMIT 5",
        "description": "Display top 5 bank names"
    },
    {
        "query": f"SELECT COUNT(*) AS Total_Banks FROM {Config.TABLE_NAME}",
        "description": "Count total number of banks"
    },
    {
        "query": f"SELECT Name, MC_USD_Billion FROM {Config.TABLE_NAME} ORDER BY MC_USD_Billion DESC LIMIT 3",
        "description": "Top 3 banks by market cap (USD)"
    }
]


# ============================================================================
# MAIN ETL PIPELINE
# ============================================================================

def run_etl_pipeline():
    """Execute the complete ETL pipeline."""
    logger.log("=" * 80)
    logger.log("ETL PIPELINE: WORLD'S LARGEST BANKS DATA")
    logger.log("=" * 80)
    logger.log(f"Start time: {datetime.now().strftime('%Y-%b-%d %H:%M:%S')}")
    logger.log("")
    
    start_time = datetime.now()
    
    try:
        # Phase 1: Extraction
        extractor = DataExtractor(Config.URL)
        df_extracted = extractor.extract_banks_data(Config.TABLE_ATTRIBUTES)
        
        if df_extracted is None or df_extracted.empty:
            logger.log("ETL PROCESS ABORTED: Extraction failed", 'ERROR')
            return False
        
        # Phase 2: Transformation
        transformer = DataTransformer(Config.CSV_EXCHANGE)
        df_transformed = transformer.transform_data(df_extracted)
        
        if df_transformed is None or df_transformed.empty:
            logger.log("ETL PROCESS ABORTED: Transformation failed", 'ERROR')
            return False
        
        # Phase 3: Load to CSV
        if not DataLoader.load_to_csv(df_transformed, Config.OUTPUT_CSV_PATH):
            logger.log("WARNING: CSV loading failed, continuing with database load...", 'WARNING')
        
        # Phase 4: Load to Database
        with DataLoader.database_connection(Config.DATABASE_NAME) as conn:
            if not DataLoader.load_to_db(df_transformed, conn, Config.TABLE_NAME):
                logger.log("ETL PROCESS ABORTED: Database loading failed", 'ERROR')
                return False
            
            # Phase 5: Verification Queries
            logger.log("")
            logger.log("=" * 60)
            logger.log("PHASE 5: VERIFICATION QUERIES")
            logger.log("=" * 60)
            
            query_executor = QueryExecutor(conn)
            for query_info in VERIFICATION_QUERIES:
                query_executor.display_query_results(
                    query_info["query"], 
                    query_info["description"]
                )
        
        # Phase 6: Data Visualization
        if VISUALIZATION_AVAILABLE:
            visualizer = DataVisualizer()
            visualizer.create_all_visualizations(df_transformed)
        else:
            logger.log("")
            logger.log("=" * 60)
            logger.log("PHASE 6: DATA VISUALIZATION")
            logger.log("=" * 60)
            logger.log("WARNING: Visualization libraries not available. Install matplotlib and seaborn to enable.")
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Final summary
        logger.log("")
        logger.log("=" * 80)
        logger.log("ETL PROCESS COMPLETED SUCCESSFULLY")
        logger.log("=" * 80)
        logger.log(f"End time: {end_time.strftime('%Y-%b-%d %H:%M:%S')}")
        logger.log(f"Execution time: {execution_time:.2f} seconds")
        logger.log(f"Banks processed: {len(df_transformed)}")
        logger.log(f"Output files:")
        logger.log(f"  - CSV: {Config.OUTPUT_CSV_PATH}")
        logger.log(f"  - Database: {Config.DATABASE_NAME}")
        if VISUALIZATION_AVAILABLE:
            logger.log(f"  - Visualizations: ./visualizations/ (5 charts generated)")
        logger.log("=" * 80)
        
        return True
        
    except Exception as e:
        logger.log(f"ETL PROCESS FAILED: {str(e)}", 'ERROR')
        import traceback
        logger.log(traceback.format_exc(), 'ERROR')
        return False


if __name__ == "__main__":
    success = run_etl_pipeline()
    sys.exit(0 if success else 1)
