from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Capital(BaseModel):
    country: str = Field(description="The country of the capital")
    capital: str = Field(description="The capital of the country")


def structured_llm():
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1", model="qwen3-30b-a3b", api_key="123"
    ).with_structured_output(Capital)

    return llm


if __name__ == "__main__":
    llm = structured_llm()
    response: BaseModel = llm.invoke("What is the capital of France?")
    print(response.model_dump_json(indent=2))
