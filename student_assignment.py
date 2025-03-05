import datetime
import chromadb
import traceback
import pandas as pd

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_collection():
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

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collections = chroma_client.list_collections()
    if "TRAVEL" not in collections:
        generate_collection()
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    return collection



def generate_hw02(question, city, store_type, start_date, end_date):
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
    query_conditions = []
    if city is not None:
        query_conditions.append({"city": {"$in": city}})
    if store_type is not None:
        query_conditions.append({"type": {"$in": store_type}})

    if start_date is not None:
        query_conditions.append({"date": {"$gte": int(start_date.timestamp())}})
    if end_date is not None:
        query_conditions.append({"date": {"$lte": int(end_date.timestamp())}})

    query_results = collection.query(
        query_texts=[question],
        n_results=10,
        where={"$and": query_conditions},
        #where_document={"$contains":"search_string"}
        include=["metadatas", "distances"]
    )
    #print(query_results)
    results = []
    delta = 1-0.8
    distances = query_results['distances'][0]
    metadatas = query_results['metadatas'][0]
    # 打印结果中的 name 字段
    for i, distance in enumerate(distances):
        if distance <= delta:
            name = metadatas[i].get("name")
            if name:  # 如果 'name' 存在，则添加到 names 列表中
                results.append(name)

    return results
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
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
    query_results = collection.query(
        query_texts=store_name,
        where={"name": store_name},  # 过滤条件
        include=["metadatas"]  # 获取元数据
    )
    #print(query_results)
    ids = query_results['ids'][0]
    metadatas = query_results['metadatas'][0]

    for metadata in metadatas:
        metadata["new_store_name"] = new_store_name

    if ids:
        collection.update(
            ids=ids,
            metadatas=metadatas,
        )
    query_results = collection.query(
        query_texts=store_name,
        where={"name": store_name},  # 过滤条件
        include=["metadatas"]  # 获取元数据
    )
    #print(query_results)

    query_conditions = []
    if city is not None:
        query_conditions.append({"city": {"$in": city}})
    if store_type is not None:
        query_conditions.append({"type": {"$in": store_type}})

    query_results = collection.query(
        query_texts=[question],
        n_results=10,
        where={"$and": query_conditions},
        # where_document={"$contains":"search_string"}
        include=["metadatas", "distances"]
    )
    # print(query_results)
    results = []
    delta = 1 - 0.8
    distances = query_results['distances'][0]
    metadatas = query_results['metadatas'][0]
    # 打印结果中的 name 字段
    for i, distance in enumerate(distances):
        if distance <= delta:
            name = metadatas[i].get("name")
            new_store_name = metadatas[i].get("new_store_name")
            if new_store_name:
                results.append(new_store_name)
            elif name:  # 如果 'name' 存在，则添加到 names 列表中
                results.append(name)

    return results
    
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
