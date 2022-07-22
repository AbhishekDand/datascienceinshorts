import streamlit as st
from happytransformer import HappyTextToText
from happytransformer import TTSettings
from PIL import Image
import numpy as np
import pandas as pd

def datascienceinshorts(blog):
    ''' This function is used to summarize the blog using distilbart model
    inputs: blog: it is a big string with the whole blog.
    ouptuts: summarized news in the form of string'''

    # initalizing the transformer
    happy_tt = HappyTextToText("DISTILBART", "sshleifer/distilbart-cnn-12-6")

    # setting up the parameters
    top_k_sampling_settings = TTSettings(do_sample=True, top_k=50, temperature=0.7, max_length=1000)

    # Finding how many sentences are there in the blog using .
    blog_new=' '.join(blog.split())
    list_index = get_index_positions(blog_new, '.')
    list_length = len(list_index)
    a = round(list_length / 3)
    b = round((2 * list_length) / 3)

    # dividing the whole blog into three parts
    part1 = blog_new[0:list_index[a]]
    part2 = blog_new[list_index[a]:list_index[b]]
    part3 = blog_new[list_index[b]:]

    # applying the transformer on all the  three parts seperately
    result1 = happy_tt.generate_text(part1, args=top_k_sampling_settings)
    result2 = happy_tt.generate_text(part2, args=top_k_sampling_settings)
    result3 = happy_tt.generate_text(part3, args=top_k_sampling_settings)

    # Appending the results
    final_result = result1.text + result2.text + result3.text
    return (final_result)


def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list


image=Image.open("Copy of PP Logo.png")
with st.sidebar:
    st.image(image,width=300)
header=st.container()
news=st.container()


with header:
    st.title('Welcome to Pandaspython!')



with news:
    st.header('Summarizer for my Data_science_in_shorts')
    col1,col2=st.columns(2)


    news = col1.text_area('Input your news here')


    col2.markdown("Here's your summary")
    if not news:
        pass
    else:
        col2.write(datascienceinshorts(news))


