import asyncio
import os
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, FSInputFile
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnableSequence

# Загрузка переменных окружения
load_dotenv()
API_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not API_TOKEN or not OPENAI_API_KEY:
    raise ValueError("TELEGRAM_TOKEN или OPENAI_API_KEY не указаны в .env")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация бота и OpenAI через LangChain
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o",
    base_url="https://api.deep-foundation.tech/v1/",
    temperature=0.3
)

# Хранение данных пользователей
user_data = {}  # {user_id: {"memory": ConversationBufferWindowMemory, "reasoning_mode": bool}}

# Клавиатура
def get_keyboard():
    buttons = [
        [KeyboardButton(text="Enable reasoning mode")],
        [KeyboardButton(text="Disable reasoning mode")]
    ]
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)

# Обычный режим
async def simple_mode(query, memory):
    prompt = PromptTemplate(
        input_variables=["history", "query"],
        template="""
        Answer the query: "{query}".  
        Consider the conversation history: {history}.  
        Determine the language of the query and respond only in that language.  
        Aim to provide the most helpful and suitable answer.
        """
    )
    chain = RunnableSequence(prompt | llm)
    response = await chain.ainvoke({"history": memory.buffer_as_str, "query": query})
    return response.content

# Режим рассуждений с конкретным финальным ответом
async def reasoning_mode(query, memory):
    # Шаг 1: Tree of Thoughts
    tot_prompt = PromptTemplate(
        input_variables=["history", "query"],
        template="""
        User query: "{query}".  
        History: {history}.  
        Generate 3 distinct reasoning paths to solve this query.  
        For each path, reason step-by-step, considering the context and aiming for a helpful solution.  
        Format each path as: "Path [N]: [reasoning]\nConclusion: [answer]".
        """
    )
    tot_chain = RunnableSequence(tot_prompt | llm)
    tot_response = await tot_chain.ainvoke({"history": memory.buffer_as_str, "query": query})
    tot_text = tot_response.content

    # Шаг 2: Самосогласованность с конкретной техникой в финале
    consistency_prompt = PromptTemplate(
        input_variables=["tot_response", "query"],
        template="""
        Here are 3 reasoning paths for the query "{query}":  
        {tot_response}  
        Evaluate each path critically: identify strengths, weaknesses, and suitability.  
        Synthesize the best ideas into a single, practical solution.  
        If the query asks for a new acting technique (e.g., "create a new acting technique"), provide a detailed, ready-to-use technique with specific exercises and instructions that is:  
        - Useful and practical for a traditional theater stage.  
        - Easy to learn and apply without complex tools or prior expertise.  
        Respond in the same language as the query.  
        Format: "Critique: [evaluation]\nSynthesized Solution: [best synthesized idea]\nFinal Answer: [detailed technique with exercises and instructions]".
        """
    )
    consistency_chain = RunnableSequence(consistency_prompt | llm)
    consistency_response = await consistency_chain.ainvoke({"tot_response": tot_text, "query": query})
    final_text = consistency_response.content

    return tot_text, final_text

# Команда /start
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    user_data[user_id] = {
        "memory": ConversationBufferWindowMemory(k=5),
        "reasoning_mode": False
    }
    await message.reply("Привет! Я бот с режимом рассуждений. Используй кнопки для управления.", reply_markup=get_keyboard())
    logging.info(f"Пользователь {user_id} запустил бота.")

# Обработка сообщений
@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    query = message.text

    if user_id not in user_data:
        user_data[user_id] = {
            "memory": ConversationBufferWindowMemory(k=5),
            "reasoning_mode": False
        }

    memory = user_data[user_id]["memory"]

    # Обработка кнопок
    if query == "Enable reasoning mode":
        user_data[user_id]["reasoning_mode"] = True
        await message.reply("Режим рассуждений включен.", reply_markup=get_keyboard())
        logging.info(f"Пользователь {user_id} включил режим рассуждений.")
        return
    elif query == "Disable reasoning mode":
        user_data[user_id]["reasoning_mode"] = False
        await message.reply("Режим рассуждений выключен.", reply_markup=get_keyboard())
        logging.info(f"Пользователь {user_id} выключил режим рассуждений.")
        return
    elif query.startswith("/"):
        return

    # Добавление запроса в память
    memory.save_context({"input": query}, {"output": ""})

    if not user_data[user_id]["reasoning_mode"]:
        # Обычный режим
        response = await simple_mode(query, memory)
        await message.reply(response)
        memory.save_context({"input": query}, {"output": response})
        logging.info(f"Пользователь {user_id} получил обычный ответ.")
    else:
        # Режим рассуждений
        tot_response, final_response = await reasoning_mode(query, memory)

        # Сохранение в MD файл
        md_content = f"# Reasoning for query: {query}\n\n"
        md_content += f"## Tree of Thoughts\n{tot_response}\n\n"
        md_content += f"## Final Evaluation\n{final_response}\n"
        file_path = f"reasoning_{user_id}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Извлечение и отправка финального ответа
        final_answer = final_response.split("Final Answer:")[1].strip() if "Final Answer:" in final_response else final_response
        await message.reply_document(FSInputFile(file_path, filename="reasoning.md"))
        await message.reply(f"Финальная техника:\n{final_answer}")
        memory.save_context({"input": query}, {"output": final_answer})
        logging.info(f"Пользователь {user_id} получил рассуждения в MD файле.")

# Запуск бота
async def main():
    logging.info("Бот запускается...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
