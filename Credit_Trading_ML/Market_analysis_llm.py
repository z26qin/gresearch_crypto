import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import requests
from dotenv import load_dotenv
from llama_cpp import Llama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

class PublicDataFetcher:
    """Class to fetch fixed income data from public sources"""
    
    def __init__(self):
        self.fred_api_key = os.getenv("FRED_API_KEY")
        if not self.fred_api_key:
            raise ValueError("FRED_API_KEY not found in environment variables")
            
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
        if not self.alpha_vantage_key:
            raise ValueError("ALPHA_VANTAGE_KEY not found in environment variables")
        
    def fetch_treasury_yields(self, start_date=None, end_date=None):
        """Fetch Treasury yield data from FRED"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Treasury yield series IDs in FRED
        series_ids = {
            '2Y': 'DGS2',
            '5Y': 'DGS5',
            '10Y': 'DGS10',
            '30Y': 'DGS30'
        }
        
        treasury_data = {}
        errors = []
        
        for tenor, series_id in series_ids.items():
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date,
                'frequency': 'd'  # Daily frequency
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract dates and values
                dates = [item['date'] for item in data['observations']]
                values = [float(item['value']) if item['value'] != '.' else None for item in data['observations']]
                
                # Store in dictionary
                treasury_data[tenor] = pd.Series(values, index=pd.DatetimeIndex(dates))
                
            except requests.exceptions.RequestException as e:
                errors.append(f"Error fetching {tenor} Treasury yield data: {e}")
        
        if errors:
            raise ValueError(f"Failed to fetch treasury yield data: {'; '.join(errors)}")
            
        # Create dataframe from series
        df = pd.DataFrame(treasury_data)
        # Forward fill missing values
        df = df.fillna(method='ffill')
        return df
    
    def fetch_corporate_spreads(self, start_date=None, end_date=None):
        """
        Fetch corporate bond spread data from FRED
        Uses ICE BofA indices and Moody's corporate bond spreads
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Corporate spread series IDs in FRED
        # Format: Rating: FRED Series ID
        series_ids = {
            'AAA': 'BAMLC0A1CAAA',  # ICE BofA AAA US Corporate Index OAS
            'AA': 'BAMLC0A2CAA',    # ICE BofA AA US Corporate Index OAS
            'A': 'BAMLC0A3CA',      # ICE BofA A US Corporate Index OAS
            'BBB': 'BAMLC0A4CBBB',  # ICE BofA BBB US Corporate Index OAS
            'BB': 'BAMLH0A1HYBB',   # ICE BofA BB US High Yield Index OAS
            'B': 'BAMLH0A2HYB',     # ICE BofA B US High Yield Index OAS
        }
        
        spread_data = {}
        errors = []
        
        for rating, series_id in series_ids.items():
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date,
                'frequency': 'd'  # Daily frequency
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract dates and values
                dates = [item['date'] for item in data['observations']]
                # Convert from basis points to percentage points for consistency
                values = [float(item['value'])/100 if item['value'] != '.' else None for item in data['observations']]
                
                # Store in dictionary
                spread_data[rating] = pd.Series(values, index=pd.DatetimeIndex(dates))
                
            except requests.exceptions.RequestException as e:
                errors.append(f"Error fetching {rating} corporate spread data: {e}")
        
        if errors:
            raise ValueError(f"Failed to fetch corporate spread data: {'; '.join(errors)}")
            
        # Create dataframe from series
        df = pd.DataFrame(spread_data)
        # Forward fill missing values
        df = df.fillna(method='ffill')
        return df
    
    def fetch_economic_indicators(self):
        """Fetch key economic indicators from FRED"""
        indicator_ids = {
            'CPI': 'CPIAUCSL',
            'Unemployment': 'UNRATE',
            'GDP_Growth': 'A191RL1Q225SBEA',
            'Fed_Rate': 'FEDFUNDS'
        }
        
        indicators = {}
        errors = []
        
        for name, series_id in indicator_ids.items():
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 4,  # Last 4 observations
                'frequency': 'm' if name != 'GDP_Growth' else 'q'  # Monthly or quarterly
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract values (most recent first)
                values = [float(item['value']) if item['value'] != '.' else None for item in data['observations']]
                # Reverse to get chronological order
                values.reverse()
                
                indicators[name] = values
                
            except requests.exceptions.RequestException as e:
                errors.append(f"Error fetching {name} data: {e}")
        
        if errors:
            raise ValueError(f"Failed to fetch economic indicators: {'; '.join(errors)}")
            
        return indicators
    
    def fetch_market_news_sentiment(self):
        """
        Fetch market news sentiment scores using Alpha Vantage News API
        """
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": "financial_markets,economy,bonds",
            "apikey": self.alpha_vantage_key,
            "limit": 50
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'feed' not in data:
                raise ValueError(f"Invalid response from Alpha Vantage: {data}")
                
            # Process news sentiment data
            sentiment_categories = {
                'Treasury_Market': [],
                'Corporate_Bonds': [],
                'Fed_Policy': []
            }
            
            for article in data['feed']:
                sentiment_score = float(article.get('overall_sentiment_score', 0))
                
                # Categorize by topics
                topics = [topic['topic'] for topic in article.get('topics', [])]
                
                if any(term in article['title'].lower() or term in article['summary'].lower() 
                       for term in ['treasury', 'government bond', 'sovereign']):
                    sentiment_categories['Treasury_Market'].append(sentiment_score)
                    
                if any(term in article['title'].lower() or term in article['summary'].lower() 
                       for term in ['corporate bond', 'credit', 'corporate debt']):
                    sentiment_categories['Corporate_Bonds'].append(sentiment_score)
                    
                if any(term in article['title'].lower() or term in article['summary'].lower() 
                       for term in ['fed', 'federal reserve', 'central bank', 'interest rate']):
                    sentiment_categories['Fed_Policy'].append(sentiment_score)
            
            # Get the 5 most recent sentiment scores for each category, or less if not enough
            for category in sentiment_categories:
                if not sentiment_categories[category]:
                    # No articles found for this category
                    sentiment_categories[category] = [0]  # Neutral if no data
                else:
                    # Take up to 5 most recent
                    sentiment_categories[category] = sentiment_categories[category][:5]
            
            return sentiment_categories
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch market news sentiment: {e}")
    
    def fetch_recent_trades(self):
        """
        Fetch recent bond trades from Alpha Vantage or other source
        """
        try:
            # This would connect to a market data API for recent trades
            # For this example, we'll simulate recent trades based on real yields
            
            # Get current Treasury yields
            treasuries = self.fetch_treasury_yields()
            last_treasury_date = treasuries.index[-1]
            recent_yields = treasuries.iloc[-1]
            
            # Create simulated recent trades based on real yields
            trades = pd.DataFrame({
                'Bond': [
                    'US Treasury 2Y', 
                    'US Treasury 5Y', 
                    'US Treasury 10Y',
                    'US Treasury 30Y',
                    'Corp AAA 5Y'  # We'll add one corporate bond
                ],
                'Price': [
                    100 - recent_yields['2Y'] * 0.5,  # Simple approximation
                    100 - recent_yields['5Y'] * 0.8,
                    100 - recent_yields['10Y'] * 1.2,
                    100 - recent_yields['30Y'] * 1.5,
                    99.5  # Made up price for corporate bond
                ],
                'Yield': [
                    recent_yields['2Y'],
                    recent_yields['5Y'],
                    recent_yields['10Y'],
                    recent_yields['30Y'],
                    recent_yields['5Y'] + 0.8  # Corporate yield = Treasury + spread
                ],
                'Volume': [
                    150000000,
                    120000000,
                    200000000,
                    80000000,
                    15000000
                ],
                'Trade_Date': [
                    last_treasury_date,
                    last_treasury_date,
                    last_treasury_date,
                    last_treasury_date,
                    last_treasury_date
                ]
            })
            
            return trades
            
        except Exception as e:
            raise ValueError(f"Failed to generate recent trades data: {e}")

    def find_best_credit_opportunities(self):
        """
        Analyze current yield curve and credit spreads to identify the best tickers to trade
        within each rating bucket
        """
        try:
            # 1. Get current treasury yields and corporate spreads
            treasuries = self.fetch_treasury_yields()
            spreads = self.fetch_corporate_spreads()
            
            # 2. Get representative tickers for each rating category
            rating_tickers = {
                'AAA': ['AAPL', 'MSFT', 'JNJ'],  # Apple, Microsoft, Johnson & Johnson
                'AA': ['GOOGL', 'XOM', 'PG'],    # Alphabet, Exxon Mobil, Procter & Gamble
                'A': ['INTC', 'KO', 'PEP'],      # Intel, Coca-Cola, PepsiCo
                'BBB': ['T', 'VZ', 'GM'],        # AT&T, Verizon, General Motors
                'BB': ['F', 'DISH', 'CCL'],      # Ford, Dish Network, Carnival
                'B': ['AMC', 'NCLH', 'M']        # AMC, Norwegian Cruise, Macy's
            }
            
            # 3. Get ticker-specific data using Alpha Vantage
            ticker_data = {}
            for rating, tickers in rating_tickers.items():
                ticker_data[rating] = self._fetch_bond_data(tickers, rating)
            
            # 4. Calculate yield curve metrics
            yield_curve = self._analyze_yield_curve(treasuries)
            
            # 5. Find best opportunities based on current conditions
            opportunities = self._identify_opportunities(ticker_data, spreads, yield_curve)
            
            return opportunities
            
        except Exception as e:
            raise ValueError(f"Failed to analyze credit opportunities: {e}")
    
    def _fetch_bond_data(self, tickers, rating):
        """Fetch specific bond data for representative tickers"""
        ticker_bonds = []
        
        for ticker in tickers:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker,
                'apikey': self.alpha_vantage_key
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                company_data = response.json()
                
                # Now get a recent bond price if available using a second call
                params['function'] = 'GLOBAL_QUOTE'
                stock_response = requests.get(url, params=params)
                stock_response.raise_for_status()
                stock_data = stock_response.json()
                
                # Create a synthetic bond based on company fundamentals
                synthetic_bond = {
                    'ticker': ticker,
                    'rating': rating,
                    'company_name': company_data.get('Name', ticker),
                    'sector': company_data.get('Sector', 'Unknown'),
                    'stock_price': float(stock_data.get('Global Quote', {}).get('05. price', 0)),
                    'pe_ratio': float(company_data.get('PERatio', 0)),
                    'debt_to_equity': float(company_data.get('DebtToEquityRatio', 0)),
                    'market_cap': float(company_data.get('MarketCapitalization', 0)),
                    # Synthetic bond metrics
                    'yield': None,  # Will calculate based on treasury + spread
                    'duration': None,  # Will calculate based on maturity
                    'z_spread': None,  # Will calculate as deviation from average spread
                    'liquidity_score': None  # Will calculate based on market cap
                }
                
                ticker_bonds.append(synthetic_bond)
                
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                print(f"Warning: Error fetching data for {ticker}: {e}")
        
        return ticker_bonds
    
    def _analyze_yield_curve(self, treasuries):
        """Analyze the current yield curve shape and dynamics"""
        # Get most recent treasury yields
        recent_yields = treasuries.iloc[-1]
        
        # Calculate key curve metrics
        slope_2_10 = recent_yields['10Y'] - recent_yields['2Y']
        slope_5_30 = recent_yields['30Y'] - recent_yields['5Y']
        belly_richness = recent_yields['5Y'] - (recent_yields['2Y'] + recent_yields['10Y'])/2
        
        # Determine curve regime
        if slope_2_10 < -0.1:
            regime = "INVERTED"
        elif slope_2_10 < 0.5:
            regime = "FLAT"
        elif slope_2_10 < 1.5:
            regime = "NORMAL"
        else:
            regime = "STEEP"
            
        # Calculate historical percentile of slope (would be better with more historical data)
        all_slopes = treasuries['10Y'] - treasuries['2Y']
        percentile_2_10 = (all_slopes < slope_2_10).mean() * 100
        
        return {
            'slope_2_10': slope_2_10,
            'slope_5_30': slope_5_30,
            'belly_richness': belly_richness,
            'regime': regime,
            'percentile_2_10': percentile_2_10,
            'recent_yields': recent_yields
        }
    
    def _identify_opportunities(self, ticker_data, spreads, yield_curve):
        """
        Identify the best opportunities based on yield curve, spreads,
        and individual bond characteristics
        """
        # Get most recent spreads
        recent_spreads = spreads.iloc[-1]
        
        # Calculate historical percentile of spreads
        spread_percentiles = {}
        for rating in spreads.columns:
            all_spreads = spreads[rating]
            spread_percentiles[rating] = (all_spreads > recent_spreads[rating]).mean() * 100
        
        # Calculate metrics for each bond and find best opportunities
        best_opportunities = {}
        
        for rating, bonds in ticker_data.items():
            if not bonds:
                best_opportunities[rating] = None
                continue
                
            for bond in bonds:
                # Calculate synthetic bond metrics
                tenor = 5  # Assume 5-year bonds for simplicity
                bond['yield'] = yield_curve['recent_yields'][f'{tenor}Y'] + recent_spreads[rating]
                bond['duration'] = 4.5  # Approximate duration for a 5-year corporate bond
                
                # Spread Z-score: how cheap/rich is this rating vs history?
                # Positive means cheap (wider spreads than normal)
                bond['z_spread'] = (recent_spreads[rating] - spreads[rating].mean()) / spreads[rating].std()
                
                # Liquidity score (1-10) based on market cap
                if bond['market_cap'] > 1e12:  # > $1 trillion
                    bond['liquidity_score'] = 10
                elif bond['market_cap'] > 5e11:  # > $500 billion
                    bond['liquidity_score'] = 9
                elif bond['market_cap'] > 1e11:  # > $100 billion
                    bond['liquidity_score'] = 8
                elif bond['market_cap'] > 5e10:  # > $50 billion
                    bond['liquidity_score'] = 7
                elif bond['market_cap'] > 1e10:  # > $10 billion
                    bond['liquidity_score'] = 6
                elif bond['market_cap'] > 5e9:  # > $5 billion
                    bond['liquidity_score'] = 5
                elif bond['market_cap'] > 1e9:  # > $1 billion
                    bond['liquidity_score'] = 4
                elif bond['market_cap'] > 5e8:  # > $500 million
                    bond['liquidity_score'] = 3
                elif bond['market_cap'] > 1e8:  # > $100 million
                    bond['liquidity_score'] = 2
                else:
                    bond['liquidity_score'] = 1
                
                # Value score - higher is better value
                bond['value_score'] = bond['z_spread'] + (bond['liquidity_score'] / 10)
            
            # Find the best opportunity in this rating category
            if bonds:
                best_bond = max(bonds, key=lambda x: x['value_score'])
                
                # Add curve positioning recommendation
                if yield_curve['regime'] == "INVERTED":
                    best_bond['duration_strategy'] = "UNDERWEIGHT" if rating in ['AAA', 'AA'] else "NEUTRAL"
                    best_bond['curve_positioning'] = "Front-end (1-3Y)"
                elif yield_curve['regime'] == "FLAT":
                    best_bond['duration_strategy'] = "NEUTRAL"
                    best_bond['curve_positioning'] = "Belly (5-7Y)"
                elif yield_curve['regime'] == "STEEP":
                    best_bond['duration_strategy'] = "OVERWEIGHT" if rating in ['A', 'BBB', 'BB', 'B'] else "NEUTRAL"
                    best_bond['curve_positioning'] = "Long-end (10Y+)"
                else:
                    best_bond['duration_strategy'] = "NEUTRAL"
                    best_bond['curve_positioning'] = "Barbell (2Y and 10Y)"
                
                # Spread dynamics
                best_bond['spread_percentile'] = spread_percentiles[rating]
                best_bond['spread_stance'] = "WIDE" if best_bond['spread_percentile'] < 25 else "TIGHT" if best_bond['spread_percentile'] > 75 else "FAIR"
                
                # Trading recommendation
                if best_bond['spread_stance'] == "WIDE" and best_bond['z_spread'] > 0.5:
                    best_bond['recommendation'] = "STRONG BUY"
                elif best_bond['spread_stance'] == "WIDE":
                    best_bond['recommendation'] = "BUY"
                elif best_bond['spread_stance'] == "TIGHT" and best_bond['z_spread'] < -0.5:
                    best_bond['recommendation'] = "STRONG SELL"
                elif best_bond['spread_stance'] == "TIGHT":
                    best_bond['recommendation'] = "SELL"
                else:
                    best_bond['recommendation'] = "HOLD"
                
                best_opportunities[rating] = best_bond
        
        return {
            'opportunities': best_opportunities,
            'yield_curve': yield_curve,
            'spread_percentiles': spread_percentiles
        }
    
    def get_market_data(self):
        """Compile all fetched data into the format expected by the analysis engine"""
        treasuries = self.fetch_treasury_yields()
        spreads = self.fetch_corporate_spreads()
        indicators = self.fetch_economic_indicators()
        news_sentiment = self.fetch_market_news_sentiment()
        recent_trades = self.fetch_recent_trades()
        
        # Package all data together
        market_data = {
            'treasuries': treasuries,
            'spreads': spreads,
            'indicators': indicators,
            'news_sentiment': news_sentiment,
            'recent_trades': recent_trades
        }
        
        # Add credit opportunities analysis
        try:
            credit_opportunities = self.find_best_credit_opportunities()
            market_data['credit_opportunities'] = credit_opportunities
        except Exception as e:
            print(f"Warning: Could not complete credit opportunities analysis: {e}")
            market_data['credit_opportunities'] = None
        
        return market_data

# Market Analysis LLM Class
class MarketAnalysisLLM:
    def __init__(self, model_path="models/llama-2-7b-chat.gguf"):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window
            n_threads=4   # Number of CPU threads to use
        )
        
        self.system_prompt = """
        You are an expert fixed income market analyst with deep knowledge of bond markets,
        yield curves, credit analysis, and macroeconomic factors. Analyze the provided market data
        and provide insights, risks, and potential trading strategies.
        
        Your analysis should include:
        1. Yield curve analysis - shape, movements, and implications
        2. Credit spread dynamics - widening/tightening and sector analysis
        3. Economic indicators impact on fixed income
        4. Market sentiment interpretation
        5. Trading recommendations with clear rationale
        
        Be specific, data-driven, and provide actionable insights for fixed income traders.
        """
        
    def _format_data(self, market_data):
        """Format market data into a readable string format for the LLM"""
        output = "# FIXED INCOME MARKET DATA ANALYSIS\n\n"
        
        # Treasury yields
        output += "## Treasury Yields (Most Recent)\n"
        recent_date = market_data['treasuries'].index[-1].strftime('%Y-%m-%d')
        treasuries = market_data['treasuries'].iloc[-1].to_dict()
        output += f"Date: {recent_date}\n"
        for tenor, yield_val in treasuries.items():
            output += f"{tenor}: {yield_val:.2f}%\n"
        
        # Treasury yield changes
        output += "\n## Treasury Yield Changes (1 Month)\n"
        yield_changes = market_data['treasuries'].iloc[-1] - market_data['treasuries'].iloc[0]
        for tenor, change in yield_changes.items():
            direction = "up" if change > 0 else "down"
            output += f"{tenor}: {change:.2f}% ({direction})\n"
        
        # Credit spreads
        output += "\n## Corporate Bond Spreads (Most Recent)\n"
        spreads = market_data['spreads'].iloc[-1].to_dict()
        for rating, spread in spreads.items():
            output += f"{rating}: +{spread:.2f}%\n"
        
        # Spread changes
        output += "\n## Corporate Spread Changes (1 Month)\n"
        spread_changes = market_data['spreads'].iloc[-1] - market_data['spreads'].iloc[0]
        for rating, change in spread_changes.items():
            direction = "widened" if change > 0 else "tightened"
            output += f"{rating}: {abs(change):.2f}% ({direction})\n"
        
        # Economic indicators
        output += "\n## Economic Indicators (Last 4 Months)\n"
        for indicator, values in market_data['indicators'].items():
            output += f"{indicator}: {', '.join([str(v) for v in values])}\n"
        
        # Recent trades
        output += "\n## Recent Bond Trades\n"
        for _, row in market_data['recent_trades'].iterrows():
            output += f"Bond: {row['Bond']}, Price: {row['Price']}, Yield: {row['Yield']}%, Volume: ${row['Volume']/1000000}M\n"
        
        # News sentiment
        output += "\n## Market Sentiment (Last 5 Days)\n"
        for segment, scores in market_data['news_sentiment'].items():
            avg_score = sum(scores) / len(scores)
            sentiment = "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "neutral"
            output += f"{segment}: {sentiment} ({avg_score:.2f})\n"
            
        return output
        
    def analyze_market(self, market_data):
        """Analyze market data and generate insights"""
        formatted_data = self._format_data(market_data)
        
        prompt = f"{self.system_prompt}\n\nPlease analyze the following fixed income market data:\n\n{formatted_data}"
        
        response = self.llm(
            prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=["Human:", "Assistant:"]
        )
        
        return response['choices'][0]['text'].strip()
    
    def generate_trade_recommendations(self, market_data, portfolio_context=None):
        """Generate specific trade recommendations"""
        formatted_data = self._format_data(market_data)
        
        portfolio_info = ""
        if portfolio_context:
            portfolio_info = f"\n\nCurrent portfolio information:\n{portfolio_context}"
        
        recommendation_prompt = """
        Based on the market data provided, generate 3-5 specific trade recommendations for a fixed income portfolio.
        For each recommendation:
        1. Specify the exact instrument (including tenor and credit quality)
        2. Indicate buy/sell recommendation
        3. Suggested size/allocation
        4. Risk level (low/medium/high)
        5. Clear rationale tied to market data
        6. Expected return and time horizon
        
        Format as a structured list with clear headers.
        """
        
        prompt = f"{self.system_prompt}\n\nPlease review the following market data:{portfolio_info}\n\n{formatted_data}\n\n{recommendation_prompt}"
        
        response = self.llm(
            prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=["Human:", "Assistant:"]
        )
        
        return response['choices'][0]['text'].strip()

    def analyze_credit_opportunities(self, opportunities_data):
        """Generate targeted analysis of credit opportunities"""
        if opportunities_data is None:
            return "Credit opportunities analysis not available due to data retrieval issues."
        
        yield_curve = opportunities_data['yield_curve']
        opportunities = opportunities_data['opportunities']
        
        # Format opportunities data for the LLM
        formatted_data = f"# CREDIT TRADING OPPORTUNITIES ANALYSIS\n\n"
        
        # Yield curve regime
        formatted_data += f"## Yield Curve Analysis\n"
        formatted_data += f"Current Regime: {yield_curve['regime']}\n"
        formatted_data += f"2Y-10Y Slope: {yield_curve['slope_2_10']:.2f}% (Percentile: {yield_curve['percentile_2_10']:.0f}%)\n"
        formatted_data += f"5Y-30Y Slope: {yield_curve['slope_5_30']:.2f}%\n"
        formatted_data += f"Belly Richness: {yield_curve['belly_richness']:.2f}%\n\n"
        
        # Best opportunities by rating
        formatted_data += f"## Best Opportunities by Rating\n\n"
        
        for rating, bond in opportunities.items():
            if bond is None:
                formatted_data += f"### {rating}: No valid opportunities found\n\n"
                continue
                
            formatted_data += f"### {rating}: {bond['ticker']} ({bond['company_name']})\n"
            formatted_data += f"Sector: {bond['sector']}\n"
            formatted_data += f"Recommendation: {bond['recommendation']}\n"
            formatted_data += f"Yield: {bond['yield']:.2f}%\n"
            formatted_data += f"Z-Spread: {bond['z_spread']:.2f} (Percentile: {bond['spread_percentile']:.0f}%)\n"
            formatted_data += f"Duration Strategy: {bond['duration_strategy']}\n"
            formatted_data += f"Curve Positioning: {bond['curve_positioning']}\n"
            formatted_data += f"Liquidity Score: {bond['liquidity_score']}/10\n\n"
        
        # Create prompt for credit opportunities analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", f"""
            Please review the following credit trading opportunities analysis and provide:
            1. A detailed assessment of each opportunity
            2. Tactical recommendations on timing and position sizing
            3. Key risks to monitor for each position
            4. Alternative trades to consider
            
            Analysis data:
            {formatted_data}
            """)
        ])
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(market_data=formatted_data)
        return response

# Modified main function to use the data fetcher with API-only data
def main():
    print("Fetching fixed income market data from APIs...")
    try:
        data_fetcher = PublicDataFetcher()
        market_data = data_fetcher.get_market_data()
        
        print("Initializing Market Analysis LLM...")
        analyzer = MarketAnalysisLLM()
        
        print("Running market analysis...")
        analysis = analyzer.analyze_market(market_data)
        print("\n" + "="*80 + "\nMARKET ANALYSIS\n" + "="*80)
        print(analysis)
        
        print("\nGenerating trade recommendations...")
        portfolio_context = """
        Current portfolio composition:
        - US Treasuries: 40% (duration: 5.2 years)
        - Investment Grade Corporate: 35% (avg rating: A-)
        - High Yield Corporate: 15% (avg rating: BB)
        - Agency MBS: 10%
        """
        recommendations = analyzer.generate_trade_recommendations(market_data, portfolio_context)
        print("\n" + "="*80 + "\nTRADE RECOMMENDATIONS\n" + "="*80)
        print(recommendations)
        
        if market_data.get('credit_opportunities'):
            print("\nAnalyzing credit opportunities by rating bucket...")
            credit_analysis = analyzer.analyze_credit_opportunities(market_data['credit_opportunities'])
            print("\n" + "="*80 + "\nCREDIT OPPORTUNITIES ANALYSIS\n" + "="*80)
            print(credit_analysis)
        
        # Plot yield curve
        plt.figure(figsize=(10, 6))
        last_day = market_data['treasuries'].iloc[-1]
        first_day = market_data['treasuries'].iloc[0]
        tenors = [2, 5, 10, 30]
        
        plt.plot(tenors, [last_day['2Y'], last_day['5Y'], last_day['10Y'], last_day['30Y']], 'b-o', label='Current')
        plt.plot(tenors, [first_day['2Y'], first_day['5Y'], first_day['10Y'], first_day['30Y']], 'r--o', label='30 Days Ago')
        
        plt.title('Treasury Yield Curve')
        plt.xlabel('Tenor (Years)')
        plt.ylabel('Yield (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig('yield_curve.png')
        print("\nYield curve chart saved as 'yield_curve.png'")
        
        # If credit opportunities data is available, plot spread vs rating
        if market_data.get('credit_opportunities'):
            plt.figure(figsize=(10, 6))
            
            # Extract ratings and spreads
            ratings = []
            spreads = []
            tickers = []
            
            for rating, bond in market_data['credit_opportunities']['opportunities'].items():
                if bond is not None:
                    ratings.append(rating)
                    spread = market_data['spreads'].iloc[-1][rating]
                    spreads.append(spread)
                    tickers.append(bond['ticker'])
            
            # Plot spreads by rating
            plt.bar(ratings, spreads)
            plt.title('Current Credit Spreads by Rating')
            plt.xlabel('Credit Rating')
            plt.ylabel('Spread (%)')
            
            # Add ticker labels
            for i, ticker in enumerate(tickers):
                plt.text(i, spreads[i] + 0.05, ticker, ha='center')
            
            plt.grid(True, axis='y')
            plt.savefig('credit_spreads.png')
            print("Credit spreads chart saved as 'credit_spreads.png'")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("Could not complete analysis due to data retrieval failure.")
        print("Please check your API keys and internet connection.")
        
if __name__ == "__main__":
    main()