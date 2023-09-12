from langchain.llms import OpenAI
import json
import random
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import openai
from openai.embeddings_utils import cosine_similarity, get_embedding
load_dotenv("KEYS.env")
os.environ['OPENAI_API_KEY'] = os.getenv("api_key")
openai.api_key = os.getenv("api_key")
llm = OpenAI(temperature = 0.81)

with open('search_word_list.txt') as f:
    data = f.read()
search_word_list = json.loads(data)

with open('ai_tool_desc.txt') as f1:
    data1 = f1.read()
ai_tool_desc = json.loads(data1)

df = pd.read_csv('words.csv')

#df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
#df.to_csv('word_embeddings.csv')

df = pd.read_csv('word_embeddings.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

search_word = input("Enter search word: ")
search_word_vector = get_embedding(search_word, engine="text-embedding-ada-002")


df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_word_vector))
df.sort_values(by=['similarities'], ascending=False, inplace=True,ignore_index=True)
term1 = df.loc[0].at['text']
term2 = df.loc[1].at['text']



pname = PromptTemplate(
   input_variables = ['num','term_desc'],
    template = "Please make this explanation of an AI tool better, double check the grammar and writing. (the closer this {num} is to 10, the longer and more complicated I want it, and the closer {num} is to 0, the shorter and simpler I want it): {term_desc}"
)
num = input("on a scale from 1-10 how much do you know about using tech? ")
random.shuffle(search_word_list[str(term1)])
term_desc = str(ai_tool_desc[str(search_word_list[str(term1)][0])])
print('\n',str(search_word_list[str(term1)][0]),'\n',llm(pname.format(num = str(num),term_desc = term_desc)),'\n')

term_desc = str(ai_tool_desc[str(search_word_list[str(term1)][1])])
print('\n',str(search_word_list[str(term1)][1]),'\n',llm(pname.format(num = str(num),term_desc = term_desc)),'\n')

term_desc = str(ai_tool_desc[str(search_word_list[str(term2)][0])])
print('\n',str(search_word_list[str(term2)][0]),'\n',llm(pname.format(num = str(num),term_desc = term_desc)),'\n')


#print(pname.format(num = str(num),term_desc =str(ai_tool_desc[str(search_word_list[str(term1)][0])])))