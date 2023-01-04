import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import tensorflow as tf
import joblib

model = tf.keras.models.load_model('model/model.h5')

actual_lables = {'anger':0,'fear':1,'joy':2,'sadness':3,'neutral':4,'surprise':5,'shame':6,'disgust':7}
actual_lables = {v: k for k, v in actual_lables.items()}


def predict_emotion(data, model):
    vect_repr = [one_hot(words,5000)for words in [data]]
    docs = pad_sequences(vect_repr,padding='pre',maxlen=50)

    prediction = np.argmax(model.predict(docs))
    print(prediction)
    prediction_prob = model.predict(docs)
    print(prediction_prob)
    prediction_class = actual_lables[prediction]

    return prediction_class, prediction_prob

emotions_emoji_dict = {"anger":"😠","fear":"😨😱", "joy":"😂", "sadness":"😔", "neutral":"😐", "surprise":"😮" , "shame":"😳", "disgust":"🤮"}

def main():
    st.title("Emotion Detection")
    
    with st.form(key='emotion-form'):
        text = st.text_area("Type here")
        submit = st.form_submit_button(label='Submit')
    
    if submit:
        col1,col2 = st.columns(2)

        predictions, predictions_probability = predict_emotion(text, model)
        
        with col1:
            st.success("Original Text")
            st.write(text)

            st.success("Prediction")
            emoji_icons = emotions_emoji_dict[predictions]
            st.write("{}:{}".format(predictions,emoji_icons))

        with col2:

            proba_df = pd.DataFrame(predictions_probability,columns=actual_lables.values())
            proba_df_ = proba_df.T.reset_index()
            proba_df_.columns = ["emotions","probability"]
            # st.write(proba_df_)

            fig = alt.Chart(proba_df_).mark_bar().encode(x='emotions',y='probability',color='emotions')
            st.altair_chart(fig,use_container_width=True)

if __name__ == "__main__":
    main()