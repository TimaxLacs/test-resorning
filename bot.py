import asyncio
import os
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, FSInputFile
from openai import OpenAI

# Загрузка переменных окружения
load_dotenv()
API_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not API_TOKEN or not OPENAI_API_KEY:
    raise ValueError("TELEGRAM_TOKEN или OPENAI_API_KEY не указаны в .env")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация бота и OpenAI
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
openai = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deep-foundation.tech/v1/")

# Хранение данных пользователей
user_data = {}  # {user_id: {"history": [], "reasoning_mode": False}}

# Клавиатура
def get_keyboard():
    buttons = [
        [KeyboardButton(text="Enable reasoning mode")],
        [KeyboardButton(text="Disable reasoning mode")]
    ]
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)

# Обычный вызов OpenAI
async def get_openai_response(messages):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Ошибка OpenAI: {e}")
        return "Error processing request / Ошибка при обработке запроса"

# Режим рассуждений
async def reasoning_mode(query, history):
    # Этап решения с акцентом на полезность и пригодность
    solve_prompt = f"""
    You are an AI assistant. The user requested: "{query}".  
    Consider the conversation history: {history[-5:]}, including past reasoning, for context.  
    Determine the language of the query and respond only in that language.  
    Reason step by step and expanded, andprovide your solution for the request, regardless of its type (question, task, reflection, etc.).  
    Aim to give the most helpful and suitable answer, not just a correct one, by adapting it to the user's likely needs and context.  
    End with a conclusion in the format: "Conclusion: [your answer]".
    """
    solve_response = await get_openai_response([{"role": "user", "content": solve_prompt}])

    # Этап проверки с самокритикой
    verify_prompt = f"""
    Verify the reasoning: "{solve_response}" for the query "{query}".  
    Consider the conversation history: including past reasoning, for context.  
    Respond in the same language as the query.  
    First, criticize the solution: identify weaknesses, assumptions, or areas where the answer can be made more helpful or suitable for the user's needs.  try to deny it completely. showing the most untenable sides of the given solution. but be structured and consistent.
    become a complete antagonist of all the proposed ideas
    Format: "Critique: [your critique]\n".
    """
    verify_response = await get_openai_response([{"role": "user", "content": verify_prompt}]) 

    improvement_prompt = f"""
    Verify the reasoning: "{solve_response}" and also criticism {verify_response} for the query "{query}".  
    Consider the conversation history: including past reasoning, for context.  
    Respond in the same language as the query.  
    Based on the criticism, try to bypass these points as much as possible by improving your ideas. It is necessary that all critical points are bypassed in one way or another or new solutions are invented that bypass them. 
    also try to provide additional upgrades that would make your answer even better. the answer needs to be the best possible
    Finally, provide a verified answer that incorporates these improvements. 
    Format: "Workaround:[your workaround]\nImprovement: [your improvements]\nVerified Answer:[verified answer]\n".
    """
    
    improvement_response = await get_openai_response([{"role": "user", "content": improvement_prompt}])

    # Этап синтеза с учетом улучшений
    synthesis_prompt = f"""
    Combine the solution "{solve_response}" and the verified answer "{verify_response}"and also "{improvement_response}" into a final answer for the query "{query}".  
    Consider the conversation history:  including past reasoning, for consistency.  
    Respond in the same language as the query.  
    Provide the final answer in the exact format requested by the user (e.g., list, paragraph, table), based on the structure and intent of the query "{query}".  
    Ensure the answer is both correct and maximally helpful, incorporating improvements from the verification step.  
    Format: "Final Answer: [final answer]".
    """
    final_response = await get_openai_response([{"role": "user", "content": synthesis_prompt}])

    return solve_response, verify_response, final_response

# Команда /start
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    user_data[user_id] = {"history": [], "reasoning_mode": False}
    await message.reply("Привет! Я бот с режимом рассуждений. Используй кнопки для управления / Hello! I'm a bot with reasoning mode. Use buttons to control.", reply_markup=get_keyboard())
    logging.info(f"Пользователь {user_id} запустил бота.")

# Обработка сообщений
@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    query = message.text

    if user_id not in user_data:
        user_data[user_id] = {"history": [], "reasoning_mode": False}

    history = user_data[user_id]["history"]

    # Обработка кнопок
    if query == "Enable reasoning mode":
        user_data[user_id]["reasoning_mode"] = True
        await message.reply("Режим рассуждений включен / Reasoning mode enabled.", reply_markup=get_keyboard())
        logging.info(f"Пользователь {user_id} включил режим рассуждений.")
        return
    elif query == "Disable reasoning mode":
        user_data[user_id]["reasoning_mode"] = False
        await message.reply("Режим рассуждений выключен / Reasoning mode disabled.", reply_markup=get_keyboard())
        logging.info(f"Пользователь {user_id} выключил режим рассуждений.")
        return
    elif query.startswith("/"):
        return

    # Добавление запроса в историю
    history.append({"role": "user", "content": query})

    if not user_data[user_id]["reasoning_mode"]:
        # Обычный режим
        prompt = f"""
        Answer the query: "{query}".  
        Consider the conversation history: {history[-5:]}.  
        Determine the language of the query and respond only in that language.  
        Aim to provide the most helpful and suitable answer, not just a correct one.
        """
        response = await get_openai_response([{"role": "user", "content": prompt}])
        await message.reply(response)
        history.append({"role": "assistant", "content": response})
        logging.info(f"Пользователь {user_id} получил обычный ответ.")
    else:
        # Режим рассуждений
        solve_response, verify_response, final_response = await reasoning_mode(query, history)

        # Сохранение в MD файл
        md_content = f"# Reasoning for query: {query}\n\n"
        md_content += f"## Solution\n{solve_response}\n\n"
        md_content += f"## Verification\n{verify_response}\n\n"
        md_content += f"## Final Answer\n{final_response}\n"
        file_path = f"reasoning_{user_id}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Отправка файла и итогового ответа
        await message.reply_document(FSInputFile(file_path, filename="reasoning.md"))
        final_answer = final_response.split("Final Answer:")[1].strip() if "Final Answer:" in final_response else final_response
        await message.reply(f"Итоговый ответ / Final Answer:\n{final_answer}")
        history.append({"role": "assistant", "content": final_answer})
        logging.info(f"Пользователь {user_id} получил рассуждения в MD файле.")

    # Ограничение истории
    if len(history) > 10:
        history[:] = history[-10:]

# Запуск бота
async def main():
    logging.info("Бот запускается...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
