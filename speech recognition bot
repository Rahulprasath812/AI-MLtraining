import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("E:\My Training -AI&ML\House Prices.csv")
df = df.drop(['ID','Date','zipcode','Floors','Lat','Long'], axis=1)

x = df[['Sqft_living', 'Bedrooms', 'Bathrooms' ]]
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#mertrics
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

#actual vs predicted
plt.figure(figsize=(10, 6)) 
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')  
plt.title('Actual vs Predicted House Prices')
plt.show()

#residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)   
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

#feature correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

#sqft_living vs price with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='Sqft_living', y='Price', data=df, scatter_kws={'alpha':0.5})
plt.title('Sqft Living vs Price with Regression Line')  
plt.xlabel('Sqft Living')
plt.ylabel('Price')
plt.show()

y_pred = model.predict(x_test)

# Step 6: Evaluate
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

new_house = pd.DataFrame({'Sqft_living': [5000], 'Bedrooms': [8], 'Bathrooms': [4]})
predicted_price = model.predict(new_house)

# Result
print("Predicted Price for new house:", predicted_price[0])

import speech_recognition as sr
import pandas as pd
import re
import pyttsx3
from sklearn.linear_model import LinearRegression

# Sample data to train
data = pd.DataFrame({
    'Size': [2000, 1500, 1800, 2200],
    'Bedrooms': [3, 2, 3, 4],
    'Age': [10, 15, 5, 8],
    'Price': [500000, 300000, 400000, 550000]
})

X = data[['Size', 'Bedrooms', 'Age']]
y = data['Price']

model = LinearRegression()
model.fit(X, y)

# Speak output
engine = pyttsx3.init('sapi5')

def speak(text):
    print("Bot:", text)
    engine.say(text)
    engine.runAndWait()

# Speech to text
recognizer = sr.Recognizer()

try:
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)

        text = recognizer.recognize_google(audio)
        print("You said:", text)

        # Flexible patterns for number extraction
        size = re.search(r'(\d+)\s*(size|square feet|sqft|square)?', text)
        bedrooms = re.search(r'(\d+)\s*(bedroom|bedrooms)?', text)
        age = re.search(r'(\d+)\s*(age|year|years)?', text)

        if size and bedrooms and age:
            size_val = int(size.group(1))
            bedrooms_val = int(bedrooms.group(1))
            age_val = int(age.group(1))

            input_data = pd.DataFrame({'Size': [size_val], 'Bedrooms': [bedrooms_val], 'Age': [age_val]})
            predicted_price = model.predict(input_data)[0]
            speak(f"Estimated house price is {round(predicted_price)} rupees.")
        else:
            speak("Sorry, I could not understand size, bedrooms, and age from what you said.")

except sr.UnknownValueError:
    speak("Sorry, I could not understand your voice.")
except sr.RequestError:
    speak("Sorry, there was an issue with the speech recognition service.")
except Exception as e:
    speak(f"Error: {e}")


