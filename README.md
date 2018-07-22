# DeepLearning
This repo is to focus on some of my work in deep learning .


## Project 1 : Simple Q&A bot which will  answer all questions of python related , management related , and banking related , from the excel sheet .

##########################################Q&A chatBot 

1. Software Used :  Python3.5 , Anaconda distribution .
2. Methodology : Doc2vec Model .

3. Attachments :
            PYTHON FILES AND MODEL: 
                a) doc2vec_model.py file (This file will  be used to  make  the doc2vec model )
                b) predicted_answers.py file ( This file is for data  processing  and codes to take user input ,for answering)
                c) doc2vec.model (This  is the  model file build using "doc2vec_model.py"  file . 


            DATASETS ATTACHED :
                a) dataset.csv (This file  contains all raw data taken from inernet )
                b) preprocessed.csv ( This fie will be preprocessed  file after  removing stopwords , uppercases to lower case converted in "question" feature .
                c) The Dataset contains 3  features ,  questions , type , answers .
                d) The features are devided into 3 classes , python , management , banking , based questions .


4. Execution : 
                a)Just put all the above attachments into same directory , change the path in predicted_answers.py file of your directory . 
                b) From command Prompt: python F://model//execute//predicted_answers.py 
                c) From Pycharm : just load the directory , execute the main function .
                d) The bot will ask you to enter question .
                e) After execution , it will ask you to press 1 or 2 .
                f) If pressed 1 , it will  give you  the  answer for the question asked . 
                g) If pressed 2 , it will  shows 5 top  matching questions , answers , ratios .
