import datetime
import chromadb
import traceback
import pandas as pd

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    # 1. 读取 CSV 文件
    csv_file = 'COA_OpenData.csv'  # 替换为你的 CSV 文件路径
    df = pd.read_csv(csv_file)

    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        # 提取 metadata 信息
        metadata = {
            "file_name": csv_file,
            "name": row['Name'],  # 店家名称
            "type": row['Type'],  # 店家类型
            "address": row['Address'],  # 店家地址
            "tel": row['Tel'],  # 店家电话
            "city": row['City'],  # 店家城市
            "town": row['Town'],  # 店家城镇
            "date": int(datetime.datetime.strptime(row['CreateDate'], "%Y-%m-%d").timestamp())  # 将 CreateDate 转换为时间戳（秒）
        }
        documents.append(row['HostWords'])
        metadatas.append(metadata)
        ids.append(str(row['ID']))

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    return collection
