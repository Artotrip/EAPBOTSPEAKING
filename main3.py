import os
import asyncio
import logging
import ffmpeg
import json
import re
from datetime import datetime

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, InputFile
import openai

# Для Google Drive API
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ─── ЗАГРУЗКА КОНФИГА И ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ─────────────────────────────────
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")  # ID папки на Google Диске

if not TELEGRAM_TOKEN:
    raise RuntimeError("Переменная окружения TELEGRAM_TOKEN не установлена")
if not OPENAI_API_KEY:
    raise RuntimeError("Переменная окружения OPENAI_API_KEY не установлена")
if not GOOGLE_SERVICE_ACCOUNT_JSON:
    raise RuntimeError("Переменная окружения GOOGLE_SERVICE_ACCOUNT_JSON не установлена")
if not GOOGLE_DRIVE_FOLDER_ID:
    raise RuntimeError("Переменная окружения GOOGLE_DRIVE_FOLDER_ID не установлена")

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
openai.api_key = OPENAI_API_KEY

# ─── ПАПКИ ДЛЯ ЛОКАЛЬНЫХ ФАЙЛОВ ──────────────────────────────────────────────
AUDIO_DIR = "voice_records_mp3"
os.makedirs(AUDIO_DIR, exist_ok=True)
TEXT_DIR = "text_records"
os.makedirs(TEXT_DIR, exist_ok=True)

LOG_FILE = "records.json"
TELEGRAM_MESSAGE_LIMIT = 4000  # примерно 4096 символов

# ─── GOOGLE DRIVE: ФУНКЦИИ ДЛЯ ЗАГРУЗКИ И ОБНОВЛЕНИЯ ФАЙЛОВ ──────────────────────────
def build_drive_service():
    """
    Строит сервис Google Drive, используя JSON ключ сервисного аккаунта из переменной окружения.
    """
    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    service = build("drive", "v3", credentials=credentials)
    return service

def find_file_on_drive(service, filename, parent_folder_id):
    """
    Ищет файл filename в указанной папке parent_folder_id.
    Возвращает fileId, если найден, иначе None.
    """
    query = f"name = '{filename}' and '{parent_folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get("files", [])
    if items:
        return items[0]["id"]
    return None

def upload_file_to_gdrive(filepath, parent_folder_id=None, is_log=False):
    """
    Загружает или обновляет файл на Google Диске.
    Если is_log=True и файл с таким именем уже есть в папке, обновляет его (files.update без поля parents).
    Иначе создаёт новый (files.create с указанием parents).
    Возвращает ID загруженного/обновлённого файла.
    """
    service = build_drive_service()
    filename = os.path.basename(filepath)

    if is_log:
        # Если это лог (records.json), попытаемся найти уже существующий файл
        existing_id = find_file_on_drive(service, filename, parent_folder_id)
    else:
        existing_id = None

    # Мета-данные: при создании обязательно указываем parents;
    # при обновлении (existing_id) не передаём поле parents вовсе.
    media = MediaFileUpload(filepath, resumable=True)

    if existing_id:
        # Обновляем существующий файл: только содержимое (media_body),
        # без изменения parents
        updated = service.files().update(
            fileId=existing_id,
            media_body=media,
            fields="id"
        ).execute()
        logging.info(f"Updated '{filename}' on Google Drive (ID={existing_id})")
        return updated.get("id")
    else:
        # Создаём новый файл, указывая parents
        file_metadata = {"name": filename}
        if parent_folder_id:
            file_metadata["parents"] = [parent_folder_id]
        created = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id"
        ).execute()
        logging.info(f"Uploaded new '{filename}' to Google Drive (ID={created.get('id')})")
        return created.get("id")

# ─── СИСТЕМНЫЙ ПРОМПТ ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Standardized Oral Language Assessment System Using ChatGPT-4o:
You are an automated oral language assessment system designed for uniform evaluation of academic English graduate students' oral monologue responses. Each response must contain exactly 10–12 complete sentences. Responses with fewer than 10 sentences automatically receive a volume score of 0. Pronunciation and Intonation are NOT assessed in this model.

All output and commentary must be in Russian, except for direct error examples and corrected English sentences, which must remain in English.

Assessment Structure:
Overall Score (2–5)
Aggregate score representing the holistic quality of the response based on criteria below.

Aspect Evaluation (each scored from 0 to 5)
Lexical and Grammatical Accuracy:
Criteria: Accuracy и appropriate complexity of grammar; precision и appropriateness of academic vocabulary.
Methodology: Automated syntactic parsing и lexical frequency analysis (Industrial Engineering: NLP-based parsing и computational linguistics).

Coherence и Cohesion:
Criteria: Logical flow of information; clear introduction, development, и conclusion; effective use of linking devices.
Methodology: Text cohesion algorithms, discourse structure analysis (Industrial Engineering: NLP coherence modeling).

Fluency и Spontaneity:
Criteria: Smooth, uninterrupted speech; minimal hesitations, repetitions, или corrections.
Methodology: Automated temporal analysis, hesitation frequency analysis (Industrial Engineering: real-time processing algorithms и temporal analytics).

Argumentation и Critical Thinking:
Criteria: Logical argumentation; effective use of examples и evidence; acknowledgment и consideration of alternative viewpoints.
Methodology: Automated reasoning analysis, semantic content evaluation (Industrial Engineering: AI-driven argumentation analysis и semantic evaluation models).

Aspect-specific Comments
Clearly articulate specific errors, citing examples directly from the student's monologue.

General Recommendation
Concise summary highlighting the student's strengths и prioritized recommendations for improvement.

List of Errors
Provide a comprehensive list of all detected grammatical и lexical errors, along with corrected versions и the topic on which the mistake was made. All examples must be quoted in English.

Practice Exercises
For each aspect where errors are detected, generate specific test-style exercises tailored to the student's mistakes. Avoid general advice. The model must create concrete grammar или vocabulary tests relevant to the identified issues. Don't write answers to the tests.

Theoretical Information (if required)
Concise theoretical background provided when the error suggests fundamental conceptual gaps (e.g., list of cohesive devices или grammar structures).
"""

# ─── ОБНОВЛЁННЫЕ FEW-SHOT ПРИМЕРЫ ──────────────────────────────────────────────
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
    "◉ international cooperating → international cooperation (Неверная форма существительного)\n"
    "◉ solving science problems → solving scientific problems (Некорректное прилагательное)\n"
    "◉ Many scientist must → Many scientists must (Ошибка в числе существительного)\n"
    "◉ it is global language → it is a global language (Пропущенный артикль)\n"
    "◉ not share outside → are not shared internationally (Ошибка в глагольной форме)\n"
    "◉ can helping scientists → can help scientists (Ошибка в форме глагола после модального)\n"
    "◉ language barrier making difficult to understanding → language barrier makes it difficult to understand (Ошибка в построении сложного оборота)\n"
    "◉ express complicated idea → express complicated ideas (Ошибка в числе существительного)\n"
    "◉ learning foreign language help scientist → learning a foreign language helps scientists (Ошибка в артикле, числе и согласовании)\n"
    "◉ translation technology solve language problem → translation technology solves the language problem (Ошибка в форме глагола)\n"
    "◉ solving of science issues → solving scientific issues (Лишний предлог + неправильное прилагательное)\n\n"
    "Практические упражнения:\n"
    "Упражнение 1: Выберите правильный вариант глагола (форма после модального):\n"
    "  Cooperation in international groups can scientists.\n"
    "  a) helping   b) help   c) helps\n\n"
    "Упражнение 2: Вставьте артикль и исправьте форму слова:\n"
    "  It is  global language.\n"
    "  Many share их ideas.\n"
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
    "Well, it can be shocking to many, но I stopped using “smartphone” - a finely made, shiny, metallic “thing”, "
    "about 5-inch tall and 3-inch wide - which I bought at least 3 years ago due to some “popular uprising” within "
    "the ranks of my immediate family members, who claimed that I could never become a “smart person” if I didn’t own a smartphone. "
    "So, after being fed up с их “constant nagging”, I finally decided to go to a smartphone store in my home town one day "
    "и offered them a “bundle” of my hard-earned money to buy a smartphone (well, that thing was darn expensive - I can tell you that). "
    "Now, on second thought, it was не только because of the “pushing и nagging” что I finally decided to buy that nice little technological wonder "
    "но also потому что it would allow мне to watch videos, receive emails и browse social media on the go. "
    "But, then, a few months ago, technology fatigue struck me as I got bored of using it too much когда I could have gone outdoors with friends. "
    "Besides, the device was so fragile что it would break если dropped. So, one day I told myself что had had enough of this smartphone thing, "
    "а that was the story of terminating моей relationship with that technology."
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
    "◉ “smartphone” - a finely made, shiny, metallic “thing” → “a smartphone – a compact, metallic device” (Избыточная разговорность)\n"
    "◉ constant nagging → persistent pressure или insistence (Разговорный стиль)\n"
    "◉ that thing was darn expensive → the device was considerably expensive (Сленг)\n"
    "◉ pushing и nagging → external social pressure (Повтор, разговорность)\n"
    "◉ hang out with my friends → spend time socially (Снижение академичности)\n"
    "◉ that damn thing → that fragile device (Сленг)\n"
    "◉ magic spell which was continuously being released → influence it exerted on my attention (Художественная метафора)\n"
    "◉ save some money because I was spending just too much money → reduce expenses due to high internet consumption (Тавтология)\n\n"
    "Практические упражнения:\n"
    "Упражнение 1: Найдите сленговые выражения и замените их на академические.\n"
    "Упражнение 2: Перепишите предложения в академическом стиле:\n"
    "  - That smartphone thing was a shiny little “magic box.”\n"
    "  - My family kept pushing и nagging me to buy it.\n"
    "  - I wanted to hang out вместо using it.\n"
    "Упражнение 3: Избегайте повторов и тавтологии:\n"
    "  Rewrite: “I stopped using it because I was spending too much money using the internet на that technology.”\n\n"
    "Теоретическая справка:\n"
    "- Формальный регистр: избегать выражений типа darn, damn, thing, hang out, magic spell.\n"
    "- Синонимы: darn expensive → considerably expensive; hang out → socialize; thing → device.\n"
)

# ─── ФУНКЦИЯ ОЦЕНИВАНИЯ ТЕКСТА ─────────────────────────────────────────────────────────────
async def assess_text(text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EXAMPLE_1_INPUT},
        {"role": "assistant", "content": EXAMPLE_1_OUTPUT},
        {"role": "user", "content": EXAMPLE_2_INPUT},
        {"role": "assistant", "content": EXAMPLE_2_OUTPUT},
        {"role": "user", "content": text},
    ]
    resp = openai.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.1,
        max_tokens=5000,
    )
    return resp.choices[0].message.content.strip()

def sanitize_filename(name: str) -> str:
    """
    Убирает из строки все символы, неприемлемые в именах файлов, и заменяет пробелы на '_'
    """
    cleaned = re.sub(r"[^A-Za-z0-9А-Яа-яёЁ\s]", "", name)
    return re.sub(r"\s+", "_", cleaned).strip("_")

def log_interaction(request_text: str, response_text: str) -> None:
    """
    Записывает запрос пользователя и ответ модели в локальный JSON и обновляет этот JSON на Google Drive.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request": request_text,
        "response": response_text
    }

    # Читаем старый файл (если есть) и добавляем новую запись
    if os.path.isfile(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except (json.JSONDecodeError, IOError):
            data = []
        data.append(entry)
    else:
        data = [entry]

    # Сохраняем локально
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Обновляем (или создаём) файл на Google Drive, передавая is_log=True
    try:
        upload_file_to_gdrive(LOG_FILE, parent_folder_id=GOOGLE_DRIVE_FOLDER_ID, is_log=True)
    except Exception as e:
        logging.error(f"Не удалось обновить {LOG_FILE} на Google Drive: {e}")

async def send_long_message(message: Message, text: str):
    """
    Разбивает большой текст на фрагменты по ~4000 символов и отправляет их последовательно.
    """
    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        await message.answer(text)
        return

    chunks = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > TELEGRAM_MESSAGE_LIMIT:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)

    for chunk in chunks:
        await message.answer(chunk)

async def send_response_as_file(message: Message, text: str, base_filename: str):
    """
    Сохраняет текст в файл и отправляет его как документ.
    Затем загружает этот файл на Google Drive как новый.
    """
    timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{base_filename}_{timestamp_str}.txt"
    filepath = os.path.join(TEXT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    await message.answer_document(InputFile(filepath))

    # Загружаем как новый файл
    try:
        upload_file_to_gdrive(filepath, parent_folder_id=GOOGLE_DRIVE_FOLDER_ID, is_log=False)
    except Exception as e:
        logging.error(f"Не удалось загрузить {filepath} на Google Drive: {e}")

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

    # Временные имена
    timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    temp_oga = f"temp_{timestamp_str}.oga"
    temp_mp3 = f"temp_{timestamp_str}.mp3"

    # Скачиваем voice.oga
    await bot.download_file(fi.file_path, temp_oga)

    # Конвертируем во временный MP3
    ffmpeg.input(temp_oga).output(temp_mp3, format="mp3").run(quiet=True, overwrite_output=True)

    # Расшифровка через Whisper
    with open(temp_mp3, "rb") as audio:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1", file=audio
        ).text.strip()

    # Удаляем временный .oga
    os.remove(temp_oga)

    # Формируем финальное имя MP3 из первых 3-4 слов транскрипции
    first_words = "_".join(transcription.split()[:4])
    sanitized = sanitize_filename(first_words)
    mp3_filename = f"{sanitized}.mp3"
    mp3_path = os.path.join(AUDIO_DIR, mp3_filename)

    # Переименовываем temp MP3 в нужное имя
    os.replace(temp_mp3, mp3_path)

    # Отправляем расшифровку пользователю
    await message.answer(f"Расшифровка:\n{transcription}")

    # Загружаем MP3 на Google Drive (каждый раз создаётся новый, аудио мы не обновляем)
    try:
        upload_file_to_gdrive(mp3_path, parent_folder_id=GOOGLE_DRIVE_FOLDER_ID, is_log=False)
    except Exception as e:
        logging.error(f"Не удалось загрузить {mp3_path} на Google Drive: {e}")

    # Оценка текста через ChatGPT
    result = await assess_text(transcription)

    # Логируем запрос и ответ: сохраняем локально и обновляем records.json на Drive
    log_interaction(request_text=transcription, response_text=result)

    # Отправляем ответ модели
    if len(result) <= TELEGRAM_MESSAGE_LIMIT:
        await message.answer(result)
    else:
        await send_long_message(message, result)
        # Или отправить как файл:
        # await send_response_as_file(message, result, base_filename="response")

# ─── ХЭНДЛЕР ТЕКСТОВЫХ ───────────────────────────────────────────────────────
@dp.message(F.text)
async def handle_text(message: Message):
    await bot.send_chat_action(message.chat.id, action="typing")
    result = await assess_text(message.text)

    # Логируем запрос–ответ
    log_interaction(request_text=message.text, response_text=result)

    # Отправляем ответ модели
    if len(result) <= TELEGRAM_MESSAGE_LIMIT:
        await message.answer(result)
    else:
        await send_long_message(message, result)
        # Или:
        # await send_response_as_file(message, result, base_filename="response")

# ─── СТАРТ ПОЛЛИНГА ─────────────────────────────────────────────────────────
async def main():
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
