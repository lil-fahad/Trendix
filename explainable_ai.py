from dotenv import load_dotenv
import os
import logging
from openai import OpenAI, error

# Securely load OPENAI_API_KEY
try:
    import streamlit as st
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY", "demo")

load_dotenv()


logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logging.error('OPENAI_API_KEY is not set in environment variables.')
    raise EnvironmentError('Missing OpenAI API key.')
client = OpenAI(api_key=api_key)

def explain_prediction(symbol, context, model_name):
    """
    Generate a textual explanation for the recommendation using GPT-4.
    """
    try:
        prompt = f"""فسر للمستثمر سبب التوصية بسهم {symbol} باستخدام النموذج {model_name}.
معلومات إضافية:
{context}
الرجاء تقديم تفسير مبسط يمكن فهمه من قبل غير المتخصصين.
"""

        response = client.chat.completions.create(
            model='gpt-4',
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )

        return response.choices[0].message.content

    except error.OpenAIError as e:
        logging.error(f'OpenAI API error: {e}')
        return 'Unable to generate explanation at this time.'
    except Exception as e:
        logging.exception('Unexpected error during prediction explanation')
        return 'An error occurred while generating the explanation.'