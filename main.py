import os
import asyncio
import logging
import ffmpeg

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
import openai

# ─── ЗАГРУЗКА КОНФИГА ─────────────────────────────────────────────────────────
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TELEGRAM_TOKEN)
dp  = Dispatcher()
openai.api_key = OPENAI_API_KEY

# ─── ПОЛНЫЙ СИСТЕМНЫЙ ПРОМПТ ─────────────────────────────────────────────────

SYSTEM_PROMPT = """Standardized Oral Language Assessment System Using ChatGPT-4o:
You are an automated oral language assessment system designed for uniform evaluation of academic English graduate students' oral monologue responses. Each response must contain exactly 10–12 complete sentences. Responses with fewer than 10 sentences automatically receive a volume score of 0. Pronunciation and Intonation are NOT assessed in this model.

All output and commentary must be in Russian, except for direct error examples and corrected English sentences, which must remain in English.

Assessment Structure:
Overall Score (2–5)
Aggregate score representing the holistic quality of the response based on criteria below.

Aspect Evaluation (each scored from 0 to 5)
Lexical and Grammatical Accuracy:
Criteria: Accuracy and appropriate complexity of grammar; precision and appropriateness of academic vocabulary.
Methodology: Automated syntactic parsing and lexical frequency analysis (Industrial Engineering: NLP-based parsing and computational linguistics).

Coherence and Cohesion:
Criteria: Logical flow of information; clear introduction, development, and conclusion; effective use of linking devices.
Methodology: Text cohesion algorithms, discourse structure analysis (Industrial Engineering: NLP coherence modeling).

Fluency and Spontaneity:
Criteria: Smooth, uninterrupted speech; minimal hesitations, repetitions, or corrections.
Methodology: Automated temporal analysis, hesitation frequency analysis (Industrial Engineering: real-time processing algorithms and temporal analytics).

Argumentation and Critical Thinking:
Criteria: Logical argumentation; effective use of examples and evidence; acknowledgment and consideration of alternative viewpoints.
Methodology: Automated reasoning analysis, semantic content evaluation (Industrial Engineering: AI-driven argumentation analysis and semantic evaluation models).

Aspect-specific Comments
Clearly articulate specific errors, citing examples directly from the student's monologue.

General Recommendation
Concise summary highlighting the student's strengths and prioritized recommendations for improvement.

List of Errors
Provide a comprehensive list of all detected grammatical and lexical errors, along with corrected versions and the topic on which the mistake was made. All examples must be quoted in English.

Practice Exercises
For each aspect where errors are detected, generate specific test-style exercises tailored to the student's mistakes. Avoid general advice. The model must create concrete grammar or vocabulary tests relevant to the identified issues. Don't write answers to the tests.

Theoretical Information (if required)
Concise theoretical background provided when the error suggests fundamental conceptual gaps (e.g., list of cohesive devices or grammar structures).
"""

# ─── FEW-SHOT ПРИМЕРЫ ────────────────────────────────────────────────────────

EXAMPLE_1_INPUT = (
"Foreign language is very important for international cooperating and solving science problems. "
"Many scientist must communicate in English because it is global language. If we don’t use a foreign language, "
"many research ideas stay only inside countries and not share outside. Cooperation in international groups can helping "
"scientists find better solutions faster. However sometimes language barrier making difficult to understanding each other clearly. "
"For example, when my research group worked with scientists from France, it was hard to express complicated idea clearly. "
"Many scientists use translators, but translators sometimes make mistakes. Therefore, learning foreign language help scientist to avoid misunderstanding. "
"Although some people argue translation technology solve language problem, I disagree. It is better if we can understand each other directly. "
"Foreign languages are therefore useful for international collaboration and solving of science issues."
)
EXAMPLE_1_OUTPUT = (
"Общая оценка: 3\n"
"Ответ в целом соответствует заданной теме, содержит 11 предложений, но качество изложения страдает из-за грамматических и лексических ошибок. "
"Аргументация поверхностна. Речь связная, но научной строгости не хватает.\n\n"
"Оценка по аспектам:\n"
"1. Лексическая и грамматическая точность: 3\n"
"   Ошибки в согласовании, выборе слов и артиклях мешают восприятию и придают высказыванию неформальный оттенок.\n"
"2. Связность и логика: 3\n"
"   Идеи изложены последовательно, но переходы между предложениями иногда резкие. Используются связки (“However,” “Therefore,” “Although”), но не всегда корректно.\n"
"3. Беглость и спонтанность: 4\n"
"   Речь построена уверенно, предложения естественно следуют друг за другом, несмотря на ошибки. Повторы минимальны.\n"
"4. Аргументация и критическое мышление: 2\n"
"   Идея понятна, но примеры поверхностны, альтернативные точки зрения не рассматриваются. Студент заявляет позицию (“I disagree”), но не объясняет её.\n\n"
"Комментарии по аспектам:\n"
"- Лексика и грамматика: Используемые фразы отражают ограниченный словарный запас. Требуется усилить академичность и точность.\n"
"- Связность: Логика в целом соблюдена, но не хватает более плавных связок между примерами.\n"
"- Аргументация: Недостаточно глубокий разбор. Примеры на уровне бытового наблюдения, а не научного обоснования.\n\n"
"Общие рекомендации:\n"
"Сфокусироваться на отработке артиклей, множественного числа и форм глаголов. Углублять аргументацию, используя примеры из научной практики. Добавить больше академических выражений и конструкций, соответствующих формальному регистру.\n\n"
"Список ошибок:\n"
"| Ошибка | Исправление | Тематика |\n"
"| international cooperating | international cooperation | Неверная форма существительного |\n"
"| solving science problems | solving scientific problems | Некорректное прилагательное |\n"
"| Many scientist must | Many scientists must | Ошибка в числе существительного |\n"
"| it is global language | it is a global language | Пропущенный артикль |\n"
"| not share outside | are not shared internationally | Ошибка в глагольной форме |\n"
"| can helping scientists | can help scientists | Ошибка в форме глагола после модального |\n"
"| language barrier making difficult to understanding | language barrier makes it difficult to understand | Ошибка в построении сложного оборота |\n"
"| express complicated idea | express complicated ideas | Ошибка в числе существительного |\n"
"| learning foreign language help scientist | learning a foreign language helps scientists | Ошибка в артикле, числе и согласовании |\n"
"| translation technology solve language problem | translation technology solves the language problem | Ошибка в форме глагола |\n"
"| solving of science issues | solving scientific issues | Лишний предлог + неправильное прилагательное |\n\n"
"Практические упражнения:\n"
"Упражнение 1: Выберите правильный вариант глагола (форма после модального):\n"
"  Cooperation in international groups can scientists.\n"
"  a) helping   b) help   c) helps\n\n"
"Упражнение 2: Вставьте артикль и исправьте форму слова:\n"
"  It is  global language.\n"
"  Many share their ideas.\n"
"  Learning foreign language is useful.\n\n"
"Упражнение 3: Найдите и исправьте ошибку:\n"
"  Many scientist must communicate in English.\n"
"  Translation technology solve language problem.\n"
"  Solving of science issues is difficult.\n\n"
"Теоретическая справка:\n"
"- Согласование: Подлежащее и сказуемое должны совпадать по числу (e.g., scientists help, not scientist help).\n"
"- Модальные глаголы: После них используется инфинитив без to (can help, must learn).\n"
"- Академическая лексика: Вместо problems лучше использовать challenges, issues; вместо help — facilitate, support.\n"
)

EXAMPLE_2_INPUT = (
"Describe some technology (e.g. an app, phone, software program) that you decided to stop using. "
"Well, it can be shocking to many, but I stopped using “smartphone” - a finely made, shiny, metallic “thing”, "
"about 5-inch tall and 3-inch wide - which I bought at least 3 years ago due to some “popular uprising” within "
"the ranks of my immediate family members, who claimed that I could never become a “smart person” if I didn’t own a smartphone. "
"So, after being fed up with their “constant nagging”, I finally decided to go to a smartphone store in my home town one day "
"and offered them a “bundle” of my hard-earned money to buy a smartphone (well, that thing was darn expensive - I can tell you that). "
"Now, on second thought, it was not only because of the “pushing and nagging” that I finally decided to buy that nice little technological wonder "
"but also because it would allow me to watch videos, receive emails and browse social media on the go. "
"But, then, a few months ago, technology fatigue struck me as I got bored of using it too much when I could have gone outdoors with friends. "
"Besides, the device was so fragile that it would break if dropped. So, one day I told myself that had had enough of this smartphone thing, "
"and that was the story of terminating my relationship with that technology."
)
EXAMPLE_2_OUTPUT = (
"Общая оценка: 4\n"
"Ответ соответствует академическому формату по объему (12 предложений), обладает связной структурой и демонстрирует беглость и уверенность в изложении. Однако использование разговорной лексики и отдельных стилистических элементов снижает академичность высказывания. Грамматические ошибки минимальны, но стилистические — ощутимы.\n\n"
"Оценка по аспектам:\n"
"1. Лексическая и грамматическая точность: 3\n"
"   Плюсы:\n"
"   - Студент демонстрирует владение сложными структурами: “due to some ‘popular uprising’ within the ranks of my immediate family members”, “technology fatigue struck me”.\n"
"   - Почти отсутствуют грамматические ошибки, за исключением редких спорных форм.\n"
"   Минусы:\n"
"   - Разговорные и эмоционально окрашенные выражения снижают академический регистр (см. список ошибок ниже).\n"
"   - Использование тавтологических и громоздких конструкций (например, “that smartphone thing”, “magic spell which was continuously being released”).\n\n"
"2. Связность и логика: 5\n"
"   Плюсы:\n"
"   - Ясная структура: вступление → объяснение → причины → отказ → последствия.\n"
"   - Используются связующие элементы: “Well,” “So,” “Now, on second thought,” “Besides,” “Therefore”.\n\n"
"3. Беглость и спонтанность: 5\n"
"   Плюсы:\n"
"   - Высокая степень спонтанности, текст звучит как живой монолог.\n"
"   - Используются вводные конструкции и усложнённые синтаксические схемы.\n\n"
"4. Аргументация и критическое мышление: 3\n"
"   Плюсы:\n"
"   - Присутствует личный опыт, перечислены как плюсы, так и минусы использования технологии.\n"
"   - Упоминается экономический фактор, психологическое восприятие, удобство.\n"
"   Минусы:\n"
"   - Недостаточно рассмотрения альтернативных точек зрения (например: «несмотря на очевидные плюсы смартфона…»).\n"
"   - Отсутствует анализ последствий отказа от технологии в научной или профессиональной сфере.\n\n"
"Комментарии по аспектам:\n"
"- Лексика: насыщенная, но не всегда академически уместная. Требуется адаптация к научному стилю.\n"
"- Грамматика: незначительные огрехи, в основном — стилистические.\n"
"- Аргументация: хороший старт, но требует усиления аналитичности.\n\n"
"Список ошибок:\n"
"| Ошибка | Исправление | Тематика |\n"
"| “smartphone” - a finely made, shiny, metallic “thing” | “a smartphone – a compact, metallic device” | Избыточная разговорность |\n"
"| constant nagging | persistent pressure or insistence | Разговорный стиль |\n"
"| that thing was darn expensive | the device was considerably expensive | Сленг |\n"
"| pushing and nagging | external social pressure | Повтор, разговорность |\n"
"| hang out with my friends | spend time socially | Снижение академичности |\n"
"| that damn thing | that fragile device | Сленг |\n"
"| magic spell which was continuously being released | influence it exerted on my attention | Художественная метафора |\n"
"| save some money because I was spending just too much money | reduce expenses due to high internet consumption | Тавтология |\n\n"
"Практические упражнения:\n"
"Упражнение 1: Найдите сленговые выражения и замените их на академические.\n"
"Упражнение 2: Перепишите предложения в академическом стиле:\n"
"  - That smartphone thing was a shiny little “magic box.”\n"
"  - My family kept pushing and nagging me to buy it.\n"
"  - I wanted to hang out instead of using it.\n"
"Упражнение 3: Избегайте повторов и тавтологии:\n"
"  Rewrite: “I stopped using it because I was spending too much money using the internet on that technology.”\n\n"
"Теоретическая справка:\n"
"- Формальный регистр: избегать выражений типа darn, damn, thing, hang out, magic spell.\n"
"- Синонимы: darn expensive → considerably expensive; hang out → socialize; thing → device.\n"
)

# ─── ФУНКЦИЯ ОЦЕНКИ ─────────────────────────────────────────────────────────
async def assess_text(text: str) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": EXAMPLE_1_INPUT},
        {"role": "assistant", "content": EXAMPLE_1_OUTPUT},
        {"role": "user",      "content": EXAMPLE_2_INPUT},
        {"role": "assistant", "content": EXAMPLE_2_OUTPUT},
        {"role": "user",      "content": text},
    ]
    resp = openai.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.1,
        max_tokens=5000,
    )
    return resp.choices[0].message.content.strip()

# ─── ХЭНДЛЕР /start ─────────────────────────────────────────────────────────
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await bot.send_chat_action(message.chat.id, action="typing")
    await message.answer(
        "Привет, аспирант! Я бот для оценки устной академической речи. "
        "Отправь текст или голосовое сообщение, и я выдам расшифровку и оценку по шаблону."
    )

# ─── ХЭНДЛЕР ГОЛОСОВЫХ ───────────────────────────────────────────────────────
@dp.message(F.voice)
async def handle_voice(message: Message):
    await bot.send_chat_action(message.chat.id, action="typing")
    fi = await bot.get_file(message.voice.file_id)
    await bot.download_file(fi.file_path, "voice.oga")
    ffmpeg.input("voice.oga").output("voice.mp3", format="mp3").run(quiet=True, overwrite_output=True)

    with open("voice.mp3", "rb") as audio:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1", file=audio
        ).text.strip()

    # расшифровка
    await message.answer(f"Расшифровка:\n{transcription}")

    # оценка
    result = await assess_text(transcription)
    await message.answer(result)

# ─── ХЭНДЛЕР ТЕКСТОВЫХ ───────────────────────────────────────────────────────
@dp.message(F.text)
async def handle_text(message: Message):
    await bot.send_chat_action(message.chat.id, action="typing")
    result = await assess_text(message.text)
    await message.answer(result)

# ─── СТАРТ ПОЛЛИНГА ─────────────────────────────────────────────────────────
async def main():
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
