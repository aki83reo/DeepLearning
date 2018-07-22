#######################PACKAGES
import os
import csv
import pandas  as pd
import nltk
import gensim
from nltk.corpus import stopwords
from gensim import corpora,models,similarities
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import  difflib
os.chdir("F://python//chatbot_march_2018//general_chatbot//doc2vecmodel")   ######### CHANGE THE PATH AS PER YOUR LOCATION

##########################################################################################################
##################################### READING   CSV FILE #########################################

df=pd.read_csv("dataset.csv", encoding = "ISO-8859-1")
df=df.dropna()   ###    Drop  if any Na values will  come

############################################################################################################

############# PREPROCESSING THE QUESTIONS IN OUR  CSV #####################################################################

for  i  in range(df.__len__()):
    questions=df['questions'][i].lower()
    stop_words = set(stopwords.words('english'))
    preprocess=[]
    from nltk.tokenize import TweetTokenizer
    tknzr = TweetTokenizer()
    preprocess.append(tknzr.tokenize(questions))
    print(preprocess)
    filtered_sentence = " ".join([w for w in preprocess[0] if not w in stop_words])
    print(filtered_sentence)
    tokens = nltk.wordpunct_tokenize(filtered_sentence)
    text = nltk.Text(tokens)
    words = " ".join([w.lower() for w in text if w.isalpha()])
    print(tokens,text,words)
    df['questions'][i]=words

#############################################################################################################

#############WRITE THE PREPROCESSED  CSV  FILE INTO LOCAL DRIVE  AS NAME  preprocessed.csv

df.to_csv("preprocessed.csv")

################ BUILDING A DOC2VEC MODEL ##################################################################

            # Various parameters
            #min_count = 1   # ignore all words with total frequency lower than this. You have to set this to 1, since the sentence labels only appear once.
            #workers = 11    # Number of threads to run in parallel
            #epocs = 20  ## number of itterations
            #window =2  ## the maximum distance between the current and predicted word within a sentence.
            # vector_size=100  dimensionality of the feature vectors in output


texts = df.to_dict('records')    #### DATAFRAME TO DICTIONERY FORMAT .
documents = [TaggedDocument(text['questions'].split(), [text['answers']])  for text in texts]   ## Pass the 'questions' & 'answers'
model = gensim.models.Doc2Vec(vector_size=100, window=2, min_count=1, workers=11,alpha=0.025, min_alpha=0.025, epochs=20)
model.build_vocab(documents)
model.train(documents, epochs=model.iter, total_examples=model.corpus_count)

#### Save  the model  in local  drive #############################

model.save("doc2vec.model")

#############################################################################################################