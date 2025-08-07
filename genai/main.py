from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

# OpenAI APIキーの取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI インスタンス
app = FastAPI()

# メモリ付きのChatモデルを定義
chat_model = ChatOpenAI(
    model="gpt-4o-mini",  # 軽量モデル
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# 会話メモリ
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# プロンプトテンプレート
prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="""
以下はユーザーとAIの会話です。
会話の履歴：
{chat_history}

ユーザー: {input}
AI:"""
)

# LLMChain 作成（メモリ付き）
conversation_chain = LLMChain(
    llm=chat_model,
    prompt=prompt,
    memory=memory
)

# 入力のデータモデル
class ChatRequest(BaseModel):
    prompt: str

# ルート確認用
@app.get("/")
def read_root():
    return {"message": "LangChain連携APIが起動しています"}

# POSTエンドポイント（会話）
@app.post("/ask")
async def ask_ai(request: ChatRequest):
    user_input = request.prompt
    response = conversation_chain.run(input=user_input)
    return {"response": response}
