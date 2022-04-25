# importing all necessary modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pickle
st.title("Text search model")
st.write("By Dennis Kevogo")

# Reads 'Youtube04-Eminem.csv' file
Nutrition=pd.read_excel('JH Technical Assignment Nutrition Questions.xlsx')
comment_words = ''
stopwords = set(STOPWORDS)
st.subheader("Reading the data in pandas")
# if st.checkbox("View the data"):
st.write("We will first start by reading the data in pandas. This data contains intents and sample questions.")
st.write(Nutrition)

st.subheader("Plotting a word cloud")  
# if st.checkbox("Exploring the tweets"):
st.write("The plot below shows the words that appear in the sample questions. Larger words shows words that are most common in the questions")
# iterate through the csv file
for val in Nutrition['Sample Questions English']:
    
    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()
    
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10, repeat=False).generate(comment_words)

# plot the WordCloud image					
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.subheader("Machine learning model")
st.write("We have trained a machine learning model that will match the sample sentences with the correct intent from nutrition-related questions")
cv = CountVectorizer()
cv.fit(Nutrition['Sample Questions English'])
X = cv.transform(Nutrition['Sample Questions English'])
# le = LabelEncoder()
# le.fit(Nutrition['Intent'])
# y = le.transform(Nutrition['Intent'])
y, unique = pd.factorize(Nutrition['Intent'])

skf = KFold(n_splits=5)
skf.get_n_splits(X, y)
print(skf)
i=0

for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rf = RandomForestClassifier()
    rf.fit(X,y)
    pred = rf.predict(X)
    i=i+1
    # st.write("The accuracy score of our text search model on fold {} is {} percent".format(i, accuracy_score(y, pred) * 100))

rf_pickle = open('random_forest.pickle', 'rb')
map_pickle = open('output.pickle', 'rb')
rfc = pickle.load(rf_pickle)
unique_mapping = pickle.load(map_pickle)

# st.write(rfc)
# st.write(unique_penguin_mapping)
rf_pickle.close()
map_pickle.close()

question = st.text_input('Enter a question to search its correct intent', 'Which food should I give my baby at 7 months? ')

# st.write('The question that you have entered is {}'.format(question))
if question != None:
    
    input_data_vec = cv.transform([question])
    predictions = rf.predict(input_data_vec)
    new_pred = unique_mapping[predictions][0]

    # predictions = le.inverse_transform(predictions)
    st.write("The correct intent to the question you have entered is: ",new_pred)

    # st.write("We predicted that your penguin is of the {} species".format(pred_species))

