import streamlit as st
from datetime import date
import requests
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import tweepy
import config

#####################
# Build the sidebar
option = st.sidebar.selectbox("Select Dashboard:",("Twitter", "StockTwits", "wallstreetbets", "chart", "patterns","Prophet"))

#st.header(option)
####################
# StockTwits
if option == "StockTwits":
    st.title("StockTwits")
    sym = st.sidebar.text_input("Symbol:", "AAPL", 5)
    r = requests.get(f'https://api.stocktwits.com/api/2/streams/symbol/{sym}.json')
    data = r.json()
    
    for m in data['messages']:
        st.image(m['user']['avatar_url'])
        st.write(m['user']['username'])
        st.write(m['created_at'])
        st.write(m['body'])
        st.markdown("""---""")

####################
# Twitter
if option == "Twitter":
    st.write(tweepy.__version__)
    #client = tweepy.Client(bearer_token=config.TWITTER_BEARER_TOKEN)

    #client = tweepy.Client(
    #consumer_key=config.TWITTER_CONSUMER_KEY, consumer_secret=config.TWITTER_CONSUMER_SECRET,
    #access_token=config.TWITTER_ACCESS_TOKEN, access_token_secret=config.TWITTER_ACCESS_TOKEN_SECRET,
    #bearer_token=config.TWITTER_BEARER_TOKEN)

    #query = 'from:traderstewie -is:retweet'

    #tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=10, user_auth=True)

    #for tweet in tweets.data:
    #    st.write(tweet.text)
    #    if len(tweet.context_annotations) > 0:
    #        st.write(tweet.context_annotations)


    #auth = tweepy.OAuthHandler(config.TWITTER_CONSUMER_KEY, config.TWITTER_CONSUMER_SECRET, config.TWITTER_ACCESS_TOKEN, config.TWITTER_ACCESS_TOKEN_SECRET)
    #api = tweepy.API(auth)
    #tweets = api.user_timeline(screen_name="traderstewie")
    
    #for tweet in tweets:
    #st.write(tweets)


####################
####################
# Prophet Example
if option == "Prophet":
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title("FB's Phophet Forecaster")

    stocks = ("SPY","AAPL", "GOOG", "MSFT","NVDA", "NFLX", "TSLA")
    selected_stock = st.selectbox("Select a stock to forecast", stocks)

    n_years = st.slider("Years to forecast:", 1, 4)
    period = n_years*365

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data = load_data(selected_stock)

    st.subheader('Raw DateFrame')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    #Forecast
    df_train = data[["Date","Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    st.write('Forecast Chart')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write('Forecast Components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)





