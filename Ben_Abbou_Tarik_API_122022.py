# text preprocessing modules
from fastapi import FastAPI
import pandas as pd

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API pour les sentiments des clients",
    version="0.1"

    # load the sentiment model
LSTM_joblib = joblib.load('my_model.pkl.pkl')

# 1. Library imports
import uvicorn
from fastapi import FastAPI


# 2. Create app and model objects
app = FastAPI()
model = LSTM_joblib 

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')

    data = pd.read_csv("training.1600000.processed.noemoticon.csv", header=None, encoding='latin-1')
    data = data.drop(['id', 'date', 'query', 'user_id'], axis=1)
    df_cleaned = data[['sentiment','comment']]
    df_cleaned['sentiment']= df_cleaned['sentiment'].astype(int) 
    
    history = model1.fit(X_train, y_train,validation_data = (X_test,y_test),epochs = 5, batch_size=32)
    yhat = model1.predict(X_test)

    @app.post('/predict')
def predict_sentiment(iris: IrisSpecies):
    data = df_cleaned
    predictionn= model1.predict_sentiment(
        data['sentiment']
    )
    return {
        'prediction': prediction,
        
    }

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)