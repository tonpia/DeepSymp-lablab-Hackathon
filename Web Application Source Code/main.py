# RUN USING THIS COMMAND
# python -m streamlit run main.py

# SETUP

## Import Important Library
import openai
import pymongo
import streamlit as st

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

## Access Credentials and Secret Keys

uri = st.secrets['MONGO_URI']
openai.api_key = st.secrets['OPENAI_API_KEY']

## SETUP Global LLM

Settings.llm = OpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)

# Access Mongo DB

## Create a new client and connect to the server
client = pymongo.MongoClient(uri)

## Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. Successfully connected to MongoDB!")
except Exception as e:
    st.exception(e)
    print(e)

store = MongoDBAtlasVectorSearch(mongodb_client = client,
                                 db_name = 'DeepSymp',
                                 collection_name = 'medical-textbook',
                                 index_name = 'vector_index')
storage_context = StorageContext.from_defaults(vector_store=store)

index = VectorStoreIndex.from_vector_store(store)

# LlamaIndex HyDE + GPT Reranker + LongContextReorder

## PostProcess
node_postprocessors = []
llm = OpenAI(model="gpt-3.5-turbo-0125", temperature=0)
node_postprocessors.append(RankGPTRerank(top_n=5, llm=llm))
node_postprocessors.append(LongContextReorder())

# Web Application

st.set_page_config(
    page_title="Demo RAG DeepSymp",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/tonpia',
        'Report a bug': "https://github.com/tonpia",
        'About': """Welcome to :blue[Deep]:green[Symp+], 
                    the web application designed to predict diseases. Our mission is to 
                    enhance the accessibility of AI technology for both medical 
                    professionals and the public."""
    }
)

st.title(':stethoscope:_:blue[DEEP]:green[SYMP+]_')
st.markdown("""Welcome to :blue[Deep]:green[Symp+], 
            the web application designed to predict diseases. Our mission is to 
            enhance the accessibility of AI technology for both medical 
            professionals and the public.""")


with st.container():
    chat_input = st.chat_input('Input Your Symptom Here',
                                max_chars=1000,
                                key = 'txt_area')

with st.container():

    col1, col2 = st.columns(2, gap = 'small')

    # with col1:

    #     map_llm = {
    #         'GPT 3.5 - Fast' : "gpt-3.5-turbo-0125", 
    #         'GPT 4 - Slow, More Accurate' : "gpt-4-turbo-2024-04-09"
    #     }


    #     llm_model = st.selectbox('Model', 
    #                 ['GPT 3.5 - Fast', 'GPT 4 - Slow, More Accurate'],
    #                 key='llm_selectbox',
    #                 label_visibility = 'collapsed'
    #     )

    #     if llm_model:
    #         Settings.llm = OpenAI(model=map_llm[st.session_state.llm_selectbox], temperature=0.1)

        


    with col2:

        map_example = {
            'Example 1 - Autoimmune Disease' : 'I have experienced pericarditis in the past. I am currently feeling sensitive and sharp pain on the dorsal aspect of both wrists, as well as the palmar face of my right wrist. I also have pain in both shoulders. The intensity of the pain is an 8, and it is precisely located at an 8. The pain appeared suddenly and does not radiate to another location. I am experiencing shortness of breath and have difficulty breathing significantly. I smoke cigarettes and have high blood pressure. I have a red rash on my cheek and nose that is not swollen but larger than 1cm. The rash does not peel off, and the itching is not severe. Additionally, I have painful mouth ulcers.', 
            'Example 2 - Epiglottitis' : 'I have a sharp knife-like pain in my right tonsil, left tonsil, back of the neck, palate, and pharynx. The intensity of the pain is a 7 out of 10 and it appeared fairly fast. I do regularly take stimulant drugs and have difficulty swallowing. I am experiencing shortness of breath and have diabetes. I do drink alcohol excessively and have noticed an increase in saliva production. I also have a high pitched sound when breathing in and my voice has become hoarse. My vaccinations are up to date and I have not traveled out of the country in the last 4 weeks.', 
            'Example 3 - Anaphylaxis' : 'I have a known severe food allergy. I have been in contact with something that I am allergic to. I have a cramp and sharp pain in my flank (left side), iliac fossa (right side), and belly. The pain is intense, around a 6. The pain appeared quickly, an 8 out of 10. I feel lightheaded and dizzy, like I am about to faint. I have lesions on my skin that are pink in color, not peeling off, and swollen at a 4 out of 10 on my back of the neck, right bicep, left bicep, mouth, and right ankle. The pain caused by the rash is 0 out of 10 but the itching is very intense at 8 out of 10. I am feeling nauseous and have a swollen cheek on the right side and nose. I have noticed a high pitched sound when breathing in and wheezing when I exhale. I am more likely to develop common allergies than the general population.',
            'Example 4 - Car Accident' : "Doctor, I wanted to share something concerning. I was in a car accident recently, and initially, I didn't feel much pain after the impact. However, now, about 24 hours later, my stomach is hurting terribly. It's been gradually getting worse since the accident, and I'm starting to feel concerned about it.",
            'Example 5 - Lung Infection' : "Doctor, we've been feeling really rough lately. Both of us have had high fevers, chills, headaches, muscle aches, and just an overall feeling of exhaustion. And the coughing... it's been relentless, accompanied by this uncomfortable tightness in our chests. It all started about two weeks ago after we spent a day hiking in the mountains of Virginia. We ventured into this dusty cave, and since then, these symptoms have just been getting worse. We thought it might pass, but it's really knocking us down now.",
            'Example 6 - zh - Epiglottitis' : '我的右扁桃体、左扁桃体、颈后、上颚和咽部都有刀割般的疼痛。疼痛的强度为 7 分（满分 10 分），而且疼痛的速度相当快。我确实经常服用兴奋剂并且吞咽困难。我感到呼吸急促并且患有糖尿病。我确实饮酒过量，并且注意到唾液分泌增加。我呼吸时声音也变高，声音也变得沙哑。我已接种最新疫苗，并且过去 4 周内没有出国旅行。',
            'Example 7 - zh - Anaphylaxis' : '我有严重的食物过敏。我接触过令我过敏的东西。我的侧腹（左侧）、髂窝（右侧）和腹部出现抽筋和剧烈疼痛。疼痛很剧烈，大约是 6 分。疼痛出现得很快，满分是 8 分。我感到头晕目眩，就像快要晕倒一样。我的皮肤上有粉红色的损伤，没有剥落，脖子后面、右二头肌、左二头肌、嘴和右脚踝处有十分之四的肿胀。皮疹引起的疼痛是十分之零，但瘙痒非常剧烈，十分之八。我感到恶心，右侧脸颊和鼻子肿胀。我注意到吸气时发出高亢的声音，呼气时发出喘息声。我比一般人更有可能出现常见过敏症。',
            'Example 8 - tr - Epiglottitis' : 'sağ bademcikimde, sol bademcikimde, ensemde, damakta ve farenksimde bıçak gibi keskin bir ağrı var. Ağrının şiddeti 10 üzerinden 7 ve oldukça hızlı ortaya çıktı. Düzenli olarak uyarıcı ilaçlar alıyorum ve yutma güçlüğü çekiyorum. Nefes darlığı çekiyorum ve şeker hastasıyım. Aşırı alkol tüketiyorum ve tükürük üretimimin arttığını fark ettim. Ayrıca nefes alırken çok tiz bir ses duyuyorum ve sesim kısılıyor. Aşılarım güncel ve son 4 haftadır yurt dışına çıkmadım.',
            'Example 9 - tr - Anaphylaxis' : 'bilinen ciddi bir gıda alerjim var. Alerjim olan bir şeyle temas ettim. Yan tarafımda (sol tarafta), iliak fossada (sağ tarafta) ve karnımda kramp ve keskin bir ağrı var. Ağrı çok yoğun, 6 civarında. Ağrı hızlı bir şekilde ortaya çıktı, 10 üzerinden 8. Başım dönüyor ve bayılacakmış gibi başım dönüyor. Cildimde ensemde, sağ pazımda, sol pazımda, ağzımda ve sağ ayak bileğimde 10 üzerinden 4 oranında pembe renkte, soyulmayan ve şişmiş lezyonlar var. Kızarıklığın neden olduğu ağrı 10 üzerinden 0 ama kaşıntı 10 üzerinden 8 ile çok yoğun. Midem bulanıyor ve sağ tarafımda ve burnumda şiş bir yanağım var. Nefes alırken yüksek perdeden bir ses ve nefes verirken hırıltı fark ettim. Genel popülasyona göre yaygın alerjilere yakalanma olasılığım daha yüksektir.'
        }

        st.selectbox('Example', [
                    'Example 1 - Autoimmune Disease', 
                    'Example 2 - Epiglottitis', 
                    'Example 3 - Anaphylaxis',
                    'Example 4 - Car Accident',
                    'Example 5 - Lung Infection',
                    'Example 6 - zh - Epiglottitis',
                    'Example 7 - zh - Anaphylaxis',
                    'Example 8 - tr - Epiglottitis',
                    'Example 9 - tr - Anaphylaxis'], 
                    key = 'example_box',
                    label_visibility = 'collapsed')
        
with st.container():
    if chat_input != '':

        if chat_input != None:
            txt_to_analyze = chat_input
        else:
            txt_to_analyze = map_example[st.session_state.example_box]
        
        st.chat_message("user").write(txt_to_analyze)
        
        query_str = f"""you are a professional medical health service you have to provide three possible diseases and reasons why using bullet points as accurately as possible and based on the given context

                  ENSURE THE RESPONSE REMAINS FAITHFUL TO THE PROVIDED CONTEXT.
                  If you don’t know the answer to a question, please don’t share false information.

                  A patient presents with the following symptoms:

                  {txt_to_analyze},

                  Please provide three possible diseases/injuries and reasons using bullet points. Additionally, specify whether the patient should seek professional medical attention or opt for self-care at a pharmacy. Outline treatment options for each identified disease.

                  Ensure the response includes:
                  - Three possible diseases/injuries with reasons
                  - Whether the patient should seek medical attention or self-care
                  - Treatment options for each identified disease/injurie

                  Response Format:
                  Possible diseases based on the symptoms described:

                  Treatment for each disease/injurie:

                  Specify whether the patient should go to a doctor or pharmacy."""
        
        query_engine = index.as_query_engine(similarity_top_k=12, 
                                     node_postprocessors=node_postprocessors,
                                     streaming=True
                                     )

        hyde = HyDEQueryTransform(include_original=True)
        hyde_query_engine = TransformQueryEngine(query_engine, hyde)

        response = hyde_query_engine.query(query_str)

        # streaming response

        message_placeholder = st.empty()
        full_response = ''

        with st.spinner(text="In progress..."):
            for chunk in response.response_gen:
                full_response += chunk
                message_placeholder.chat_message("assistant").write(full_response + ' ')

        message_placeholder.chat_message("assistant").write(full_response + ' ')

        st.success('Done!')
        st.toast('Your symptoms have been Analyzed!', icon='🩺')