from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from typing import Annotated, Any, TypedDict


import pandas as pd
import logging, base64
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class State(TypedDict):
    product_id: Annotated[str, "product id"]
    product_name: Annotated[str, "product name"]
    product_company: Annotated[str, "product company"]
    model_name: Annotated[str, "model name"]
    keyword: Annotated[str, "keyword"]
    category: Annotated[str, "My custom category"]
    image_urls: Annotated[list[str], "Porceesed Image URL"]
    origin_df: Annotated[pd.DataFrame, "Orignal data frame"]


class Preprocess:
    def __call__(self, df: pd.DataFrame, model_name: str, manafacturing_company_name: str) -> pd.DataFrame:
        df = self.remove_duplicate(df)
        df = self.change_model_name(df, model_name)
        df = self.change_manafacturing_company_name(df, manafacturing_company_name)

        return df

    @staticmethod
    def remove_duplicate(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["제품번호"], inplace=False)

    @staticmethod
    def change_model_name(df: pd.DataFrame, name: str) -> pd.DataFrame:
        return Preprocess.change_value_of_column(df, "모델명", name)

    @staticmethod
    def change_manafacturing_company_name(df: pd.DataFrame, name: str) -> pd.DataFrame:
        return Preprocess.change_value_of_column(df, "제조사", name)

    @staticmethod
    def change_value_of_column(df: pd.DataFrame, key, value):
        df[key] = value

        return df


def process_product_name(state: State):
    system_prompt: str = (
        "사용자가 제공하는 제품명 에서 아래 조건들을 고려하며 완성형 명사 형태의 요소들로 수정해주세요.\n"
        "단독 문자열 형태로만 출력해주세요 ``` 사용금지. 나머지 출력은 필요없습니다.\n"
        "조건 1. 최소 꼭 필요한 요소만 남기고 지우기\n"
        "- 슈퍼그립 Cool 쿨한 장갑에서 Cool 같은 중복 의미 제거\n"
        "- 판단에 따라 쿨한도 뺄 수  있음\n"
        "조건 2. 제목을 인지하기 어려운 코드형 문자 지우기\n"
        "- _0066600 등의 코드 형태의 불필요한 내용\n"
        "조건 3. 유명한 브랜드가 들어간 명사의 경우는 해당 명사만 제거\n"
        "- 블랙야크 점퍼패딩에서 블랙야크는 한국 브랜드임\n"
        "조건 4. 제품에 대한 명확한 설명들은 남기기\n"
        "- 제품이 설명이 되는지가 중요 \n"
        "- 예를들어 그레이 프로젝터 -> 그레이는 색상에 대한 설명\n"
    )

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{product_name}")])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    output_parser = StrOutputParser()

    chain = {"product_name": RunnablePassthrough()} | prompt | llm | output_parser
    product_name = state["product_name"]
    new_product_name = chain.invoke(product_name)
    logging.info(f"New product name : {state["product_name"]} -> {new_product_name}")

    return State(product_name=new_product_name)


def process_image_url(state: State) -> State:
    from graphs.params import DOME79
    from graphs.utils import get_image_from_url
    from ultralytics.models.fastsam import FastSAMPredictor
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import torch
    import cv2
    import datetime
    import os
    import logging

    # Initialize models
    overrides = dict(
        conf=0.78,
        task="detect",
        mode="predict",
        model="FastSAM-s.pt",
        save=False,
        save_crop=False,
        imgsz=1024,
    )
    predictor = FastSAMPredictor(overrides=overrides)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Get standard images
    standard_images = []
    for url in state["image_urls"]:
        image = get_image_from_url(url)
        if image is not None:
            standard_images.append(image)

    if not standard_images:
        return State(image_urls=[None] * len(state["image_urls"]))

    # Get detailed images
    detailed_page_urls = DOME79.get_detail_image_page_url(state["product_id"])
    detailed_images = []
    for url in detailed_page_urls:
        image = get_image_from_url(url)
        if image is not None:
            detailed_images.append(image)

    # Extract images using FastSAM
    extracted_images = []
    results = predictor(detailed_images)
    for result in results:
        boxes = result.boxes.xyxy.cpu().tolist()
        for box in boxes:
            x1, x2 = int(box[0]), int(box[2])
            y1, y2 = int(box[1]), int(box[3])
            crop_obj = result.orig_img[y1:y2, x1:x2]
            crop_obj = cv2.cvtColor(crop_obj, cv2.COLOR_BGR2RGB)
            crop_obj = Image.fromarray(crop_obj, "RGB")
            extracted_images.append(crop_obj)

    if not extracted_images:
        return State(image_urls=[None] * len(state["image_urls"]))

    # Find best matching image using CLIP
    threshold_sim = 0.5
    candidates = []
    for image in extracted_images:
        inputs = clip_processor(
            text=None,
            images=standard_images + [image],
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = clip_model.get_image_features(pixel_values=inputs["pixel_values"])

        standard_features = outputs[:-1]
        compared_feature = outputs[-1]
        similarities = torch.nn.functional.cosine_similarity(standard_features, compared_feature.unsqueeze(0), dim=1)
        mean_similarity = similarities.mean().item()

        if mean_similarity >= threshold_sim:
            candidates.append([image, mean_similarity])

    if not candidates:
        return State(image_urls=[None] * len(state["image_urls"]))

    # Process best image and add watermark
    best: Image.Image = sorted(candidates, key=lambda x: x[-1])[-1][0]
    best = best.resize((1000, 1000))
    logging.info("Resizing ! ")
    scale_factor = 0.4
    margin = 10
    opacity = 0.4

    try:
        watermark_image = Image.open("./data/watermark.png")
        watermark_image = watermark_image.convert("RGBA")
        if watermark_image is None:
            raise ValueError("Failed to load watermark image")

        # Calculate watermark size based on image dimensions while maintaining aspect ratio
        watermark_ratio = watermark_image.size[0] / watermark_image.size[1]
        if best.size[0] > best.size[1]:
            new_width = int(best.size[0] * scale_factor)
            new_height = int(new_width / watermark_ratio)
        else:
            new_height = int(best.size[1] * scale_factor)
            new_width = int(new_height * watermark_ratio)

        watermark = watermark_image.resize((new_width, new_height))

        # Handle watermark transparency
        alpha_channel = watermark.split()[3]
        alpha_channel = alpha_channel.point(lambda x: int(x * opacity))
        watermark.putalpha(alpha_channel)

        # Position watermark
        x = best.width - watermark.width - margin
        y = best.height - watermark.height - margin

        # Create and apply overlay
        overlay = Image.new("RGBA", best.size, (0, 0, 0, 0))
        overlay.paste(watermark, (x, y), mask=watermark)
        logging.info("Paste watermark")

        # Composite final image
        best = best.convert("RGBA")
        logging.info(f"1. {best.size} 2. {overlay.size}")
        best = Image.alpha_composite(best, overlay)

        # Save image
        output_dir = f"outputs/{datetime.datetime.now().strftime('%Y-%m-%d')}"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{state['product_id']}.jpg")
        best = best.convert("RGB")
        best.save(save_path, quality=95)
        logging.info(f"Successfully saved image to: {save_path}")

        return State(
            image_urls=[f"https://gi.esmplus.com/lukasacc/79DM/{state['product_id']}.jpg"] * len(state["image_urls"])
        )

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return State(image_urls=[None] * len(state["image_urls"]))


def process_keyword(state: State) -> State:
    from graphs.params import NaverAPI
    import hmac, hashlib, time, requests

    hint_keywords = [state["product_name"]]
    hint_keywords = [keyword.replace(" ", "") for keyword in hint_keywords]
    timestamp = str(round(time.time() * 1000))
    message = "{}.{}.{}".format(timestamp, "GET", "/keywordstool")
    hash = hmac.new(bytes(NaverAPI.secret_key, "utf-8"), bytes(message, "utf-8"), hashlib.sha256)
    hash.hexdigest()
    signature = base64.b64encode(hash.digest())
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Timestamp": timestamp,
        "X-API-KEY": NaverAPI.api_key,
        "X-Customer": NaverAPI.customer_id,
        "X-Signature": signature,
    }

    params = {"hintKeywords": hint_keywords}
    r = requests.get(url=NaverAPI.url, params=params, headers=headers)
    logging.info(f"request : res.request")
    res = r.json()

    keywords = res["keywordList"]

    from langchain_core.documents import Document

    documents = [Document(page_content=f"{item['relKeyword']}", metadata=item) for item in keywords]
    logging.info(f"Document : {documents[0]}")

    from graphs.retriever_chain import BaseRetrieverChain
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    loader = None
    splitter = None
    embedding = OpenAIEmbeddings()
    vector_store = FAISS
    retriver = BaseRetrieverChain(
        documents=documents, loader=loader, splitter=splitter, vector_store=vector_store, embdding=embedding
    ).retriever

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an assistant for question-answering about keyword tasks.\n"
                    "Use the following pieces of retrieved context to answer the question.\n"
                    "If you don't know the answer, just say like [] (empty list)\n"
                    "Answer like ['keyword-1', 'keyword-2', ..., 'keyword-N'] (list[str]).\n"
                ),
            ),
            ("user", ("#Question: \n" "{question}\n" "#Context: \n" "{context}\n" "#Answer:")),
        ]
    )

    chain = {"context": retriver, "question": RunnablePassthrough()} | prompt | ChatOpenAI() | StrOutputParser()
    question = f"{state['product_name']}과 가장 관련된 키워드 5가지 알려줘"
    response = chain.invoke(question)
    logging.info(f"Response : {response}")
    keyword = eval(response)
    logging.info(f"Keyword : {keyword}, {type(keyword)}")

    return State(keyword=keyword)


def process_category(state: State) -> State:
    from graphs.retriever_chain import BaseRetrieverChain
    from langchain_community.document_loaders import DataFrameLoader
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    import pandas as pd

    file_path = "./data/마이카테-1.xls"
    df = pd.read_excel(file_path)

    loader = DataFrameLoader(df, page_content_column="마이카테")

    db_path = "./data/db"
    embedding = OpenAIEmbeddings()
    vector_store = FAISS
    db = BaseRetrieverChain(loader=loader, vector_store=vector_store, vector_store_path=db_path, embdding=embedding).db

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             (
    #                 "You are an assistant for question-answering about category tasks.\n"
    #                 "Use the following pieces of retrieved context to answer the question.\n"
    #                 "If you don't know the answer, just say like 'None' (empty list)\n"
    #                 "Answer only WBxxx (A category of column '마이카테') Don't including anything.\n"
    #             ),
    #         ),
    #         ("user", ("#Question: \n" "{question}\n" "#Context: \n" "{context}\n" "#Answer:")),
    #     ]
    # )

    from langchain_core.documents import Document
    import numpy as np

    # chain = {"context": retriver, "question": RunnablePassthrough()} | prompt | ChatOpenAI() | StrOutputParser()
    question = f"{state['product_name']}과 가장 관련된 마이카테 하나 골라줘."
    response: list[tuple[Document, np.float32]] = db.similarity_search_with_score(query=question, k=5)
    response.sort(key=lambda x: x[-1], reverse=True)
    logging.info(f"Response : {response}")
    category = [doc.page_content for doc, _ in response][0]
    logging.info(f"Category : {category}")

    return State(category=category)


def export_result_excel(state: State) -> State:
    import datetime

    df = state["origin_df"].loc[state["origin_df"]["제품번호"] == state["product_id"]].copy()
    # Update DataFrame with processed data
    df["상품명"] = state["product_name"]
    df["제조사"] = state["product_company"]
    df["모델명"] = state["model_name"]
    df["키워드"] = ",".join(state["keyword"])
    df["마이카테"] = state["category"]
    for i in range(1, len(state["image_urls"]) + 1):
        df[f"목록이미지{i}"] = state["image_urls"][i - 1]

    # Create output directory if it doesn't exist
    output_dir = f"outputs/excel/{datetime.datetime.now().strftime('%Y-%m-%d')}"
    os.makedirs(output_dir, exist_ok=True)

    # Save to Excel
    output_dir = f"outputs/excel/{datetime.datetime.now().strftime('%Y-%m-%d')}"
    output_path = os.path.join(output_dir, "result.xlsx")
    if not os.path.exists(output_path):
        os.makedirs(output_dir, exist_ok=True)
        df.to_excel(output_path, index=False)
        logging.info(f"Successfully exported results to: {output_path}")
    else:
        existing = pd.read_excel(output_path)

        df_appended = pd.concat([existing, df])
        df_appended.to_excel(output_path, index=False)

    return state


if __name__ == "__main__":
    from langgraph.graph import StateGraph, START, END
    from dotenv import load_dotenv
    import pandas as pd

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    input = State()
    file_path = "./data/List_20250324215443_joung6517_1.xlsx"
    df = pd.read_excel(file_path)
    preprocessor = Preprocess()
    new_df = preprocessor(
        df[:2], model_name="LO" + df["제품번호"].astype(str), manafacturing_company_name="LOTTO협력사"
    )
    # Get columns directly from the DataFrame
    result_columns = df.columns.tolist()
    example_df = pd.DataFrame(columns=result_columns)

    graph_builder = StateGraph(State)
    graph_builder.add_node("process product name", process_product_name)
    graph_builder.add_node("process image url", process_image_url)
    graph_builder.add_node("process keyword", process_keyword)
    graph_builder.add_node("process category", process_category)
    graph_builder.add_node("export result excel", export_result_excel)

    graph_builder.add_edge(START, "process product name")
    graph_builder.add_edge("process product name", "process image url")
    graph_builder.add_edge("process image url", "process keyword")
    graph_builder.add_edge("process keyword", "process category")
    graph_builder.add_edge("process category", "export result excel")
    graph_builder.add_edge("export result excel", END)

    graph = graph_builder.compile()

    from IPython.display import Image, display
    from langchain_core.runnables.graph import CurveStyle, NodeStyles, MermaidDrawMethod

    try:
        graph_image = graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
            output_file_path="graph.png",  # Save to file first
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white",
            padding=10,
        )
        display(Image(filename="graph.png"))
    except Exception as e:
        print(f"Error displaying graph: {str(e)}")

    for _, row in new_df.iterrows():
        state = State(
            product_id=row.제품번호,
            product_name=row.상품명,
            product_company=row.제조사,
            model_name=row.모델명,
            keyword=row.키워드,
            category=row.마이카테,
            image_urls=[
                row.목록이미지1,
                row.목록이미지2,
                row.목록이미지3,
                row.목록이미지4,
                row.목록이미지5,
                row.목록이미지6,
                row.목록이미지7,
            ],
            origin_df=df,
        )
        try:
            for event in graph.stream(state):
                for idx, value in enumerate(event.values()):
                    logging.info(f"Value {idx} : {value}")

        except Exception as e:
            print(str(e))
            raise
