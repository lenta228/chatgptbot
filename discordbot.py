import os
import asyncio
import logging
from dotenv import load_dotenv
import g4f
import requests
import json
import traceback
import sys
import websockets
import time
import random
import zlib
import re

# Настройка подробного логирования
logging.basicConfig(
    level=logging.DEBUG,  # Изменил уровень на DEBUG для большей детализации
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('chatgpt_bot')

# Добавляем вывод логов в консоль
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Загрузка переменных окружения
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# Вывод информации о токене (без показа полного токена)
if TOKEN:
    logger.info(f"Токен загружен (первые 5 символов: {TOKEN[:5]}...)")
else:
    logger.critical("Токен не найден! Проверьте файл .env")

# Кэш для сохранения контекста беседы с пользователями
user_conversations = {}

# Максимальное количество сообщений в истории для каждого пользователя
MAX_HISTORY = 10

# Базовый URL для API Discord
BASE_URL = "https://discord.com/api/v10"
GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"

# Константы для Gateway
DISCORD_GATEWAY_VERSION = 10
OP_DISPATCH = 0
OP_HEARTBEAT = 1
OP_IDENTIFY = 2
OP_PRESENCE_UPDATE = 3
OP_RESUME = 6
OP_RECONNECT = 7
OP_INVALID_SESSION = 9
OP_HELLO = 10
OP_HEARTBEAT_ACK = 11

# Интенты для Gateway
INTENT_GUILDS = 1 << 0
INTENT_GUILD_MEMBERS = 1 << 1
INTENT_GUILD_MESSAGES = 1 << 9
INTENT_MESSAGE_CONTENT = 1 << 15
INTENT_DIRECT_MESSAGES = 1 << 12

# Доступные провайдеры GPT
GPT_PROVIDERS = ["g4f", "you", "openai_chat", "openai", "copilot", "gemini", "anthropic"]
GPT_MODELS = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "claude-instant-1"]

# Функция для форматирования текста для Discord
def format_for_discord(text):
    """Форматирует текст для корректного отображения в Discord"""
    formatted_text = text
    
    # Обработка многострочных блоков кода - сохраняем их вместо удаления
    # Ищем блоки кода ```...``` и сохраняем их
    code_blocks = []
    code_block_regex = r'```(?:[\w]*\n)?([\s\S]*?)```'
    
    # Находим все блоки кода и запоминаем их
    code_blocks = re.findall(code_block_regex, formatted_text)
    
    # Заменяем блоки кода на плейсхолдеры
    formatted_text = re.sub(code_block_regex, '___CODE_BLOCK___', formatted_text)
    
    # Заменяем однострочные блоки кода (кроме тех, что уже в плейсхолдерах)
    formatted_text = formatted_text.replace("`", "")
    
    # Обработка заголовков и выделения текста
    lines = formatted_text.split("\n")
    for i in range(len(lines)):
        # Заголовки первого уровня
        if lines[i].startswith("# "):
            lines[i] = "**" + lines[i][2:] + "**"
        # Заголовки второго уровня
        elif lines[i].startswith("## "):
            lines[i] = "**" + lines[i][3:] + "**"
        # Заголовки третьего уровня
        elif lines[i].startswith("### "):
            lines[i] = "**" + lines[i][4:] + "**"
        # Заголовки четвертого уровня
        elif lines[i].startswith("#### "):
            lines[i] = "**" + lines[i][5:] + "**"
        # Заголовки пятого уровня
        elif lines[i].startswith("##### "):
            lines[i] = "**" + lines[i][6:] + "**"
        # Заголовки шестого уровня
        elif lines[i].startswith("###### "):
            lines[i] = "**" + lines[i][7:] + "**"
    
    formatted_text = "\n".join(lines)
    
    # Обработка ссылок Markdown [текст](ссылка) -> текст (ссылка)
    formatted_text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', formatted_text)
    
    # Убедимся, что нет лишних или незакрытых маркеров форматирования
    # Проверка на парность ** для жирного текста
    count_bold = formatted_text.count("**")
    if count_bold % 2 != 0:
        formatted_text += "**"
    
    # Проверка на парность * для курсива
    count_italic = formatted_text.count("*")
    if count_italic % 2 != 0:
        formatted_text += "*"
    
    # Проверка на парность _ для подчеркивания
    count_underline = formatted_text.count("_")
    if count_underline % 2 != 0:
        formatted_text += "_"
    
    # Проверка на парность ~~ для зачеркивания
    count_strike = formatted_text.count("~~")
    if count_strike % 2 != 0 and count_strike > 0:
        formatted_text += "~~"
    
    # Проверка на парность || для спойлеров
    count_spoiler = formatted_text.count("||")
    if count_spoiler % 2 != 0 and count_spoiler > 0:
        formatted_text += "||"
    
    # Возвращаем блоки кода обратно после форматирования
    for code_block in code_blocks:
        formatted_text = formatted_text.replace('___CODE_BLOCK___', f'```\n{code_block}\n```', 1)
    
    return formatted_text

async def generate_gpt_response(messages):
    """Функция для генерации ответа от GPT, с автоматическим перебором провайдеров при ошибке"""
    logger.debug(f"Генерация ответа GPT для сообщений: {json.dumps(messages)}")
    
    for provider in GPT_PROVIDERS:
        for model in GPT_MODELS:
            try:
                logger.debug(f"Пробую провайдера {provider} с моделью {model}")
                response = await asyncio.to_thread(
                    g4f.ChatCompletion.create,
                    model=model,
                    messages=messages,
                    provider=eval(f"g4f.Provider.{provider}") if provider != "g4f" else None
                )
                
                if response and len(response) > 0:
                    logger.debug(f"Успешный ответ от {provider}/{model}: {response[:50]}...")
                    # Применяем форматирование для Discord
                    formatted_response = format_for_discord(response)
                    return formatted_response
                else:
                    logger.warning(f"Пустой ответ от {provider}/{model}, пробую следующий...")
            except Exception as e:
                logger.warning(f"Ошибка при использовании {provider}/{model}: {str(e)}")
    
    # Если все провайдеры не сработали, используем базовый подход
    try:
        logger.debug("Пробую базовый метод без указания провайдера")
        response = await asyncio.to_thread(
            g4f.ChatCompletion.create,
            messages=messages
        )
        # Применяем форматирование для Discord
        formatted_response = format_for_discord(response)
        return formatted_response
    except Exception as e:
        logger.error(f"Все провайдеры GPT завершились с ошибкой: {str(e)}")
        raise Exception("Не удалось получить ответ от GPT после всех попыток")

class DiscordBot:
    def __init__(self, token):
        logger.debug("Инициализация бота...")
        self.token = token
        self.headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json"
        }
        self.user_id = None
        self.ws = None
        self.session_id = None
        self.sequence = None
        self.heartbeat_interval = None
        self.last_heartbeat_ack = True
        self.heartbeat_task = None
        self.zlib_buffer = bytearray()
        self.inflator = zlib.decompressobj()
        self.resume_gateway_url = None
        self.reconnect_counter = 0
        self.max_reconnect_attempts = 5

    async def get_bot_info(self):
        """Получение информации о боте"""
        logger.debug(f"Запрашиваю информацию о боте через API...")
        url = f"{BASE_URL}/users/@me"
        
        try:
            logger.debug(f"Выполняю GET-запрос к {url}")
            response = requests.get(url, headers=self.headers)
            
            logger.debug(f"Получен ответ. Код: {response.status_code}")
            logger.debug(f"Заголовки ответа: {response.headers}")
            
            if response.status_code == 200:
                data = response.json()
                self.user_id = data["id"]
                logger.info(f"Бот успешно авторизован: {data['username']}#{data.get('discriminator', '')}")
                logger.debug(f"Полные данные бота: {json.dumps(data, indent=2)}")
                return data
            else:
                logger.error(f"Ошибка при получении информации о боте: {response.status_code}")
                logger.error(f"Тело ответа: {response.text}")
                raise Exception(f"Ошибка API: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Исключение при получении информации о боте: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def send_message(self, channel_id, content, reference_message_id=None):
        """Отправка сообщения в канал"""
        url = f"{BASE_URL}/channels/{channel_id}/messages"
        
        payload = {"content": content}
        
        # Если есть ID сообщения для ответа, добавляем его
        if reference_message_id:
            payload["message_reference"] = {"message_id": reference_message_id}
        
        logger.debug(f"Отправка сообщения в канал {channel_id}. Payload: {json.dumps(payload)}")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            logger.debug(f"Ответ сервера: {response.status_code}")
            
            if response.status_code == 200 or response.status_code == 201:
                logger.debug("Сообщение успешно отправлено")
                return response.json()
            else:
                logger.error(f"Ошибка при отправке сообщения: {response.status_code}")
                logger.error(f"Тело ответа: {response.text}")
                raise Exception(f"Ошибка API: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Исключение при отправке сообщения: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def process_gpt_request(self, user_id, username, channel_id, message_id, content):
        """Обработка запроса к GPT и отправка ответа"""
        logger.debug(f"Обработка запроса от пользователя {username} (ID: {user_id})")
        logger.debug(f"Содержание запроса: {content}")
        
        try:
            # Отправляем typing-индикатор, чтобы показать, что бот печатает
            typing_url = f"{BASE_URL}/channels/{channel_id}/typing"
            try:
                requests.post(typing_url, headers=self.headers)
                logger.debug("Отправлен индикатор набора текста")
            except Exception as e:
                logger.warning(f"Не удалось отправить индикатор набора: {str(e)}")
            
            # Создаем сообщения для GPT
            messages = [{"role": "user", "content": content}]
            
            # Отправляем запрос к GPT
            logger.info(f"Отправка запроса к GPT для пользователя {username}")
            
            # Используем функцию для генерации ответа GPT с автоматическим перебором провайдеров
            gpt_response = await generate_gpt_response(messages)
            logger.debug(f"Получен ответ от GPT: {gpt_response[:100]}...")
            
            # Форматируем ответ для Discord
            logger.debug(f"Размер ответа: {len(gpt_response)} символов")
            
            # Разбиваем длинные ответы на части, если они превышают лимит Discord
            max_length = 2000
            if len(gpt_response) <= max_length:
                logger.debug("Отправка ответа целиком")
                await self.send_message(channel_id, gpt_response, message_id)
            else:
                # Разбиваем ответ на части и отправляем по частям
                logger.debug(f"Разбиваю ответ на части по {max_length} символов")
                parts = [gpt_response[i:i + max_length] for i in range(0, len(gpt_response), max_length)]
                logger.debug(f"Количество частей: {len(parts)}")
                
                for i, part in enumerate(parts):
                    if i == 0:
                        logger.debug(f"Отправка части {i+1}/{len(parts)} как ответ на сообщение")
                        await self.send_message(channel_id, part, message_id)
                    else:
                        logger.debug(f"Отправка части {i+1}/{len(parts)} как продолжение")
                        await self.send_message(channel_id, part)
                        
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send_message(channel_id, f"Произошла ошибка при обработке вашего запроса: {str(e)}", message_id)

    async def process_message(self, message_data):
        """Обработка входящего сообщения"""
        logger.debug(f"Обработка сообщения: {json.dumps(message_data)}")
        
        # Пропускаем сообщения от ботов
        if message_data.get('author', {}).get('bot', False):
            logger.debug("Сообщение от бота, пропускаю")
            return

        content = message_data.get('content', '')
        channel_id = message_data.get('channel_id')
        message_id = message_data.get('id')
        author = message_data.get('author', {})
        user_id = author.get('id')
        username = author.get('username')
        
        logger.debug(f"Сообщение от {username} (ID: {user_id}): {content}")
        
        # Проверка на сообщение из ветки форума
        thread = message_data.get('thread')
        thread_context = ""
        
        # Если сообщение из ветки форума, получаем информацию о теме
        if thread:
            try:
                # Получаем информацию о теме
                thread_url = f"{BASE_URL}/channels/{thread.get('id')}"
                thread_response = requests.get(thread_url, headers=self.headers)
                if thread_response.status_code == 200:
                    thread_data = thread_response.json()
                    thread_name = thread_data.get('name', '')
                    thread_context = f"Тема форума: {thread_name}\n"
                    logger.debug(f"Получен контекст темы: {thread_context}")
            except Exception as e:
                logger.error(f"Ошибка при получении информации о теме: {str(e)}")
        
        # Проверяем, упомянут ли бот в сообщении
        mentions = message_data.get('mentions', [])
        # Также проверяем, есть ли упоминание роли, в которой состоит бот
        mention_roles = message_data.get('mention_roles', [])
        
        # Проверяем прямое упоминание бота или роли
        is_bot_mentioned = any(mention.get('id') == self.user_id for mention in mentions)
        
        # Если есть mention_roles, нужно проверить принадлежат ли они боту
        # В данном случае просто используем любое упоминание роли как триггер
        is_role_mentioned = len(mention_roles) > 0
        
        if is_bot_mentioned or is_role_mentioned:
            logger.debug(f"Бот или его роль упомянуты в сообщении (ID бота: {self.user_id})")
            # Удаляем упоминание бота из сообщения
            cleaned_content = content
            if is_bot_mentioned:
                cleaned_content = cleaned_content.replace(f'<@{self.user_id}>', '').strip()
            
            # Удаляем упоминания ролей
            if is_role_mentioned:
                for role_id in mention_roles:
                    cleaned_content = cleaned_content.replace(f'<@&{role_id}>', '').strip()
            
            logger.debug(f"Очищенное содержание: {cleaned_content}")
            
            if not cleaned_content:
                logger.debug("Пустой запрос после упоминания бота")
                await self.send_message(channel_id, "Пожалуйста, укажите ваш вопрос после упоминания бота.", message_id)
                return
            
            # Добавляем контекст темы к запросу, если он есть
            if thread_context:
                cleaned_content = f"{thread_context}\nЗапрос пользователя: {cleaned_content}"
                
            await self.process_gpt_request(user_id, username, channel_id, message_id, cleaned_content)

    async def heartbeat(self):
        """Отправка регулярных heartbeat для поддержания соединения с Gateway"""
        try:
            logger.debug(f"Запуск задачи отправки heartbeat (интервал: {self.heartbeat_interval/1000} сек)")
            while True:
                if not self.last_heartbeat_ack:
                    logger.warning("Не получен ACK на предыдущий heartbeat, переподключение...")
                    return False
                
                # Отправка heartbeat
                payload = {
                    "op": OP_HEARTBEAT,
                    "d": self.sequence
                }
                logger.debug(f"Отправка heartbeat: seq={self.sequence}")
                await self.ws.send(json.dumps(payload))
                self.last_heartbeat_ack = False
                
                # Ждем интервал
                await asyncio.sleep(self.heartbeat_interval / 1000)
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"Соединение разорвано во время heartbeat: {e}")
            return False
        except Exception as e:
            logger.error(f"Ошибка в задаче heartbeat: {e}")
            logger.error(traceback.format_exc())
            return False

    async def identify(self):
        """Отправка идентификации бота для установления соединения с Gateway"""
        logger.debug("Отправка identify в Gateway...")
        
        # Рассчитываем интенты
        intents = (INTENT_GUILDS | INTENT_GUILD_MESSAGES | INTENT_MESSAGE_CONTENT | INTENT_DIRECT_MESSAGES)
        
        # Получаем текущее время для аптайма
        current_time = time.strftime("%H:%M:%S")
        
        payload = {
            "op": OP_IDENTIFY,
            "d": {
                "token": self.token,
                "intents": intents,
                "properties": {
                    "os": "windows",
                    "browser": "chatgpt_bot",
                    "device": "chatgpt_bot"
                },
                "presence": {
                    "activities": [
                        {
                            "name": f"time: {current_time}",
                            "type": 0
                        }
                    ],
                    "status": "online",
                    "afk": False
                }
            }
        }
        
        try:
            await self.ws.send(json.dumps(payload))
            logger.debug("Identify отправлен")
            return True
        except Exception as e:
            logger.error(f"Ошибка при отправке identify: {e}")
            return False

    async def resume(self):
        """Восстановление прерванной сессии"""
        if not self.session_id or not self.resume_gateway_url:
            logger.warning("Нет данных для resume, выполняю новое подключение")
            return False
            
        logger.debug(f"Восстановление сессии {self.session_id}...")
        
        try:
            # Подключаемся к URL восстановления
            logger.debug(f"Подключаюсь к {self.resume_gateway_url}")
            self.ws = await websockets.connect(self.resume_gateway_url)
            
            payload = {
                "op": OP_RESUME,
                "d": {
                    "token": self.token,
                    "session_id": self.session_id,
                    "seq": self.sequence
                }
            }
            
            await self.ws.send(json.dumps(payload))
            logger.debug("Запрос на resume отправлен")
            return True
        except Exception as e:
            logger.error(f"Ошибка при восстановлении сессии: {e}")
            return False

    def _unpack_data(self, data):
        """Распаковка сжатых данных из WebSocket"""
        if isinstance(data, bytes):
            self.zlib_buffer.extend(data)
            
            # Проверяем, завершен ли пакет
            if len(data) < 4 or data[-4:] != b'\x00\x00\xff\xff':
                return None
                
            # Распаковываем данные
            try:
                decompressed = self.inflator.decompress(self.zlib_buffer)
                self.zlib_buffer = bytearray()
                return json.loads(decompressed)
            except Exception as e:
                logger.error(f"Ошибка распаковки данных: {e}")
                self.zlib_buffer = bytearray()
                return None
        else:
            # Данные не сжаты
            return json.loads(data)

    async def process_gateway_event(self, data):
        """Обработка событий от Discord Gateway"""
        op = data.get('op')
        
        if op == OP_DISPATCH:
            event_type = data.get('t')
            event_data = data.get('d')
            self.sequence = data.get('s') if data.get('s') is not None else self.sequence
            
            logger.debug(f"Получено событие: {event_type}, seq={self.sequence}")
            
            if event_type == 'READY':
                self.session_id = event_data.get('session_id')
                self.resume_gateway_url = event_data.get('resume_gateway_url')
                logger.info(f"Бот готов! Сессия: {self.session_id}")
                
            elif event_type == 'MESSAGE_CREATE':
                # Обрабатываем новое сообщение
                await self.process_message(event_data)
                
        elif op == OP_HELLO:
            # Получение интервала heartbeat
            self.heartbeat_interval = data.get('d', {}).get('heartbeat_interval', 45000)
            logger.debug(f"Получен HELLO от сервера. Интервал heartbeat: {self.heartbeat_interval/1000} сек")
            
            # Запускаем heartbeat
            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
            self.heartbeat_task = asyncio.create_task(self.heartbeat())
            
            # Отправляем identify, если это новое соединение
            if not self.session_id:
                await self.identify()
            
        elif op == OP_HEARTBEAT_ACK:
            logger.debug("Получен HEARTBEAT_ACK")
            self.last_heartbeat_ack = True
            
        elif op == OP_RECONNECT:
            logger.warning("Сервер запросил переподключение (OP_RECONNECT)")
            return False
            
        elif op == OP_INVALID_SESSION:
            resumable = data.get('d', False)
            if resumable:
                logger.warning("Получен INVALID_SESSION (resumable)")
                await asyncio.sleep(random.uniform(1, 5))
                return await self.resume()
            else:
                logger.warning("Получен INVALID_SESSION (не resumable)")
                self.session_id = None
                self.sequence = None
                await asyncio.sleep(random.uniform(1, 5))
                return await self.identify()
        
        return True

    async def connect_to_gateway(self):
        """Подключение к Discord Gateway"""
        try:
            logger.info("Подключение к Discord Gateway...")
            
            self.ws = await websockets.connect(GATEWAY_URL)
            logger.debug("WebSocket соединение установлено")
            
            # Сброс состояния heartbeat
            self.last_heartbeat_ack = True
            
            while True:
                try:
                    # Получаем данные
                    data = await self.ws.recv()
                    parsed_data = self._unpack_data(data)
                    
                    if parsed_data:
                        # Обрабатываем событие
                        success = await self.process_gateway_event(parsed_data)
                        if not success:
                            logger.warning("Требуется переподключение...")
                            break
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(f"Соединение с Gateway разорвано: {e}")
                    break
            
            return False
        except Exception as e:
            logger.error(f"Ошибка при подключении к Gateway: {e}")
            logger.error(traceback.format_exc())
            return False

    async def gateway_connection_handler(self):
        """Управление подключением к Gateway с автоматическим переподключением"""
        self.reconnect_counter = 0
        
        while self.reconnect_counter < self.max_reconnect_attempts:
            if self.reconnect_counter > 0:
                backoff = min(2 ** self.reconnect_counter, 60)
                logger.info(f"Переподключение через {backoff} секунд (попытка {self.reconnect_counter}/{self.max_reconnect_attempts})...")
                await asyncio.sleep(backoff)
            
            try:
                # Проверка наличия session_id для resume
                if self.session_id and self.resume_gateway_url:
                    logger.info("Попытка восстановления сессии...")
                    success = await self.resume()
                    if not success:
                        # Если не удалось восстановить, то устанавливаем новое соединение
                        success = await self.connect_to_gateway()
                else:
                    # Новое подключение
                    success = await self.connect_to_gateway()
                    
                if success:
                    # Сброс счетчика переподключений при успешном соединении
                    self.reconnect_counter = 0
                else:
                    # Увеличиваем счетчик неудачных попыток
                    self.reconnect_counter += 1
            except Exception as e:
                logger.error(f"Исключение в обработчике подключения: {e}")
                logger.error(traceback.format_exc())
                self.reconnect_counter += 1
        
        logger.critical(f"Превышено максимальное количество попыток переподключения ({self.max_reconnect_attempts})")

    async def start(self):
        """Запуск бота"""
        try:
            logger.info("Запуск бота...")
            
            # Получаем информацию о боте
            logger.debug("Получение информации о боте...")
            try:
                bot_info = await self.get_bot_info()
                logger.info(f"Информация о боте получена: ID = {bot_info.get('id')}, Username = {bot_info.get('username')}")
            except Exception as e:
                logger.critical(f"Не удалось получить информацию о боте: {str(e)}")
                logger.critical(traceback.format_exc())
                return
            
            # Проверка серверов бота
            try:
                logger.debug("Проверка серверов бота...")
                guilds_url = f"{BASE_URL}/users/@me/guilds"
                response = requests.get(guilds_url, headers=self.headers)
                
                if response.status_code == 200:
                    guilds = response.json()
                    logger.info(f"Бот присутствует в {len(guilds)} серверах:")
                    for guild in guilds:
                        logger.info(f"  - {guild.get('name')} (ID: {guild.get('id')})")
                else:
                    logger.error(f"Не удалось получить список серверов: {response.status_code}")
                    logger.error(f"Ответ: {response.text}")
            except Exception as e:
                logger.error(f"Ошибка при проверке серверов: {str(e)}")
            
            # Подключаемся к Discord Gateway
            logger.info("Подключение к Discord Gateway...")
            await self.gateway_connection_handler()
            
        except KeyboardInterrupt:
            logger.info("Бот остановлен пользователем (Ctrl+C)")
        except Exception as e:
            logger.critical(f"Критическая ошибка: {str(e)}")
            logger.critical(traceback.format_exc())

# Запуск бота
if __name__ == "__main__":
    try:
        logger.info("Старт программы")
        
        # Проверка наличия токена
        if not TOKEN:
            logger.critical("DISCORD_TOKEN не найден. Убедитесь, что файл .env создан и содержит правильный токен.")
            sys.exit(1)
            
        logger.debug("Создание экземпляра бота")
        bot = DiscordBot(TOKEN)
        
        logger.debug("Запуск асинхронного цикла бота")
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Программа остановлена пользователем (Ctrl+C)")
    except Exception as e:
        logger.critical(f"Не удалось запустить бота: {str(e)}")
        logger.critical(traceback.format_exc())
