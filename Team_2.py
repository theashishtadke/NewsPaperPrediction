
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



'''

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.utils import np_utils
import tensorflow as tf

X = data[['content']].values.tolist()
y = data[['publication']].values.tolist()
clean_y = []
for j in y:
    clean_y.append(j[0])

clean_X, clean_y = shuffle(clean_X, clean_y, random_state = 0)
onehot= pd.get_dummies(clean_y)
target_labels = onehot.columns
Y_1 = onehot.as_matrix()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(clean_X, Y_1 , test_size = 0.2, random_state = 0)




from sklearn.feature_extraction.text import TfidfVectorizer
counter = TfidfVectorizer()
counter.fit(X_train)
counts_train = counter.transform(X_train)#transform the training data
counts_test = counter.transform(X_test)#transform the testing data

np.random.seed(1337)
nb_classes = len(np.unique(y_train))
batch_size = 64
nb_epochs = 20

#Y_train = np_utils.to_categorical(y_train, nb_classes, dtype = 'string')
input_shape = counts_train.shape[1]
model = Sequential()

model.add(Dense(1000,input_shape= (input_shape,)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(500))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(50))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes))

#model.add(Activation('softmax'))
model.add(Activation(tf.nn.softmax))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(counts_train, y_train, batch_size=batch_size, epochs=nb_epochs,verbose=1)

y_train_predclass = model.predict_classes(counts_train)


y_test_predclass = model.predict_classes(counts_test,batch_size=batch_size)

from sklearn.metrics import accuracy_score,classification_report

print ("nnDeep Neural Network - Train accuracy:"),((accuracy_score( Y_train, y_train_predclass)))

print ("nDeep Neural Network - Test accuracy:"),((accuracy_score( Y_test,y_test_predclass)))

print ("nDeep Neural Network - Train Classification Report")

print (classification_report(y_train,y_train_predclass))

print ("nDeep Neural Network - Test Classification Report")

print (classification_report(y_test,y_test_predclass))


from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')


ids = onehot.idxmax(axis=1)

onehot.dot(onehot.columns)


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)
onehotencoder = OneHotEncoder()
X_2 = onehotencoder.fit_transform(y_encoded.reshape(-1,1)).toarray()

y_train_decoded = onehotencoder.inverse_transform(y_train)

print ("nnDeep Neural Network - Train accuracy:"),((accuracy_score( y_train_decoded, y_train_predclass)))
accuracy_score( y_train_decoded, y_train_predclass)
'''