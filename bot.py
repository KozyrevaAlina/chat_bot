import time
import logging
from aiogram import Bot, Dispatcher, executor, types

from bi_encoder import get_bank_answers#, get_best_answer
from cross_encoder import re_rank
import cfg


logging.basicConfig(level=logging.INFO)

bot = Bot(token=cfg.TOKEN)
dp = Dispatcher(bot=bot)

query = []
greeting_message = ''

@dp.message_handler(commands=["start"])
async def start_handler(message: types.Message):
    global greeting_message

    user_id = message.from_user.id
    user_full_name = message.from_user.full_name

    logging.info(f'{user_id} {user_full_name} {time.asctime()}')
    greeting_message = f"Hi, {user_full_name} {cfg.MSG}!"
    query.append(greeting_message)
    await message.reply(greeting_message)
    

@dp.message_handler() 
async def echo(message: types.Message): 
    global query

    if len(query) < 5: #2 размер контекста
        query.append(message.text)
        # answer = get_best_answer(query)
        answers = get_bank_answers(query)
        answer = re_rank(query, answers)
        query.append(answer)
    else:
        query = []
        query.append(message.text)
        # answer = get_best_answer(query)
        answers = get_bank_answers(query)
        answer = re_rank(query, answers)

        query.append(answer)

    logging.info(f'{query}')
    await message.answer(answer)
   
if __name__ == "__main__": 
    executor.start_polling(dp)


