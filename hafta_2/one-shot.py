import transformers
import langchain
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

def zero_shot_classification():
    """Sıfır örnekle metin sınıflandırma"""
    user_input = input("Please write your comment\n")
    prompt =  f"""Girilen yorumun duygu tonunu aşağıdaki formatta belirle.
            Bu yorumun duygusal tonunu belirle.(pozitif, negatif, nötr):
            yorum: {user_input}
            duygusal ton: """

    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role" : "user", "content": prompt}
        ],
        max_tokens = 50,
        temperature = 0
    )

    return response.choices[0].message.content

if __name__=="__main__":
    print("==Duygu Analizi \n")
    try:
        result = zero_shot_classification()
        print(f"Sonuç : {result}\n")
    except Exception as e:
        print(f"Hata! : {e}\n")