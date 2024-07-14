import requests
import streamlit as st
import json

def main():

    st.title("Image classification")

    image = st.file_uploader("Choose an image", type=['jpg', 'jpeg'])

    if st.button("Classify!") and image is not None:
        st.image(image)
        files = {"file": image.getvalue()}
        res = requests.post("http://127.0.0.1:8000/classify", files=files).json()
        st.write(f'Class name: {res["class_name"]}, class index: {res["class_index"]}')
        # st.write(json.loads(res.text)['prediction'])

    txt = st.text_input('here')

    if st.button('send'):
        dat = {'text' : txt}
        res = requests.post("http://127.0.0.1:8000/clf_text", json=dat)
        st.write(res.json()['label'])
        st.write(res.json()['prob'])

if __name__ == '__main__':
    main()
