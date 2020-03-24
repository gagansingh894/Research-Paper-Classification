# IN ORDER TO RUN THIS YOU MIGHT NEED TO INSTALL PACKAGES IF YOU DON'T HAVE THEM ALREADY
import re
import joblib
import pandas as pd
import numpy as np
import pickle
import tqdm
import spacy
from keras.models import load_model

nlp = spacy.load("en_core_web_md") 
# IF ERROR RUN THE FOLLOWING COMMAND IN TERMINAl OR CMD  =>  python -m spacy download en_core_web_md
nlp.vocab[u"i"].is_stop = False # REMOVE FROM STOPWORDS
nlp.vocab[u"I"].is_stop = False # REMOVE FROM STOPWORDS
targetMap = {1:'fit', 0:'no fit'} # FOR MAPPING PREDICTIONS

# LOAD MODELS
model1 = joblib.load('model_renn_svc_scale_balanced.pkl')
model2 = load_model('model_renn_ann_dropout_weights.h5')

#CONFIG
m_num = 0 # SET 0 FOR SVM, 1 FOR ANN 
out_filename = "Data_Prediction_SVM.csv"


# LOAD NEW DATA 
# ENTER THE PATH OF YOUR FILE WHICH CONTAINS THE TITLE.
# ENSURE HEADER FOR TITLE IS PRESENT IN THE FILE. THE NAME OF HEADER SHOULD BE 'title'  
PATH = r"30K titles_processed.xlsx" #ENTER YOUR PATH INSIDE THE QUOTES
p_df = pd.read_excel(PATH)
# USE THE BELOW COMMAND IF YOU ARE USING A .csv FILE AS INPUT
# p_df = pd.read_csv(PATH, names=['title']) 

# PREPARE THE NEW DATA - APPLY THE SAME LOGICS WHICH WERE APPLIED ON THE TRAINING DATA
def prepare_predict(df):
    processed= []
    X= []

    for i in tqdm.tqdm(range(len(df))):
        doc = nlp(df['title'][i])
        txt = [re.sub('[^a-zA-Z0-9]', ' ',token.lemma_, flags=re.IGNORECASE).strip().lower() for token in doc if not token.is_stop and not token.is_punct]
        txt = ' '.join(txt)
        processed.append(txt)

    df['processed'] = processed
    df['processed'] = df['processed'].str.replace('pron', 'i', case=False)
    # df.drop('title', inplace=True, axis=1)
    del processed, i, txt
    
    for i in tqdm.tqdm(range(len(df['processed']))):
        X.append(nlp(df['processed'][i]).vector.reshape(-1,300))

    X = np.asarray(X).reshape(-1,300)    
    
    if m_num == 0: 
        return model1.predict(X)
    elif m_num == 1:
        return model2.predict_classes(X)

def savetoexcel():
    pred = prepare_predict(p_df)
    p_df['predictions'] = pred
    p_df['predictions'] = p_df['predictions'].map(targetMap)
    p_df.to_csv(out_filename, index=False)

savetoexcel()