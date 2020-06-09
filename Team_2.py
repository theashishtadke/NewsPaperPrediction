
import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
from sklearn.utils import shuffle
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
from sklearn.linear_model import LogisticRegression
from nltk.corpus import wordnet

train_data = pd.read_csv('reviews_train.txt', sep="\t", header=None)
test_data = pd.read_csv('reviews_test.txt', sep="\t", header=None)


#data=pd.read_csv("articles1.csv")

import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

#X = data[['content']].values.tolist()
#y = data[['publication']].values.tolist()

X_train = train_data.iloc[:, 0]
Y_train = train_data.iloc[:, 1]

X_test = test_data.iloc[:, 0]
Y_test = test_data.iloc[:, 1]

clean_X_train = []
for i in X_train:
    clean_X_train.append(clean_text(i))
    
clean_X_test = []
for i in X_test:
    clean_X_test.append(clean_text(i))

#y = pd.DataFrame(data=data.iloc[:, 3].values)
#count = 0 
#clean_X = []
#clean_y = []
#for i in X:
#    clean_X.append(clean_text(i[0]))
#    #count +=1
#    #print(count)    
#for j in y:
#    clean_y.append(j[0])
#
#clean_X, clean_y = shuffle(clean_X, clean_y, random_state = 0)

#
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(clean_X, clean_y , test_size = 0.18, random_state = 0)
from sklearn.feature_extraction.text import TfidfVectorizer
counter = TfidfVectorizer()
counter.fit(X_train)
counts_train = counter.transform(clean_X_train)#transform the training data
counts_test = counter.transform(clean_X_test)#transform the testing data
clf= LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',verbose =1, n_jobs = -1)
#clf = MultinomialNB()
#import xgboost as xgb
#clf = xgb.XGBClassifier(max_depth = 3, n_estimators = 300, learning_rate = 0.05)

#train all classifier on the same datasets
clf.fit(counts_train,Y_train)

#use hard voting to predict (majority voting)
pred=clf.predict(counts_test)

#print accuracy
print (accuracy_score(pred,Y_test))



