from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os
import logging
import traceback
from typing import Dict, List, Optional
from datetime import datetime
import threading
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # Разрешаем все origins для тестирования
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"]
    }
})

class OpenRouterClient:
    def __init__(self):
        """Инициализация клиента OpenRouter"""
        # Конфигурация моделей с правильными API ключами
        self.models = {
            "qwen": {
                "id": "qwen/qwen2.5-72b-instruct",
                "api_key": "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",
                "role": "latex_specialist",
                "description": "LaTeX и математические вычисления",
                "system_prompt": "Ты специалист по LaTeX и математике."
            },
            "mistral": {
                "id": "mistral/mistral-nemo",
                "api_key": "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",
                "role": "calculator",
                "description": "Вычисления и анализ",
                "system_prompt": "Ты математический калькулятор."
            }
        }
        
        # Словарь для хранения контекста разговоров
        self.conversation_contexts = {}
        self.context_lock = threading.Lock()
        
        # Инициализация клиентов для каждой модели
        self.clients = {}
        for model_key, config in self.models.items():
            try:
                self.clients[model_key] = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=config["api_key"]
                )
                logger.info(f"Инициализирован клиент для модели {model_key}")
            except Exception as e:
                logger.error(f"Ошибка инициализации клиента {model_key}: {e}")
    
    def get_or_create_context(self, session_id: str) -> Dict[str, List[Dict[str, str]]]:
        """Получение или создание контекста разговора"""
        with self.context_lock:
            if session_id not in self.conversation_contexts:
                self.conversation_contexts[session_id] = {}
                for model_key in self.models:
                    self.conversation_contexts[session_id][model_key] = []
                    # Добавляем системное сообщение
                    system_prompt = self.models[model_key].get("system_prompt", "")
                    if system_prompt:
                        self.conversation_contexts[session_id][model_key].append({
                            "role": "system",
                            "content": system_prompt
                        })
            return self.conversation_contexts[session_id]
    
    def chat_with_model(self, model_key: str, message: str, session_id: str) -> str:
        """Синхронная отправка сообщения конкретной модели"""
        if model_key not in self.models:
            raise ValueError(f"Неизвестная модель: {model_key}")
        
        if model_key not in self.clients:
            raise ValueError(f"Клиент для модели {model_key} не инициализирован")
        
        model_config = self.models[model_key]
        context = self.get_or_create_context(session_id)
        
        # Подготавливаем сообщения
        messages = context[model_key].copy()
        messages.append({"role": "user", "content": message})
        
        try:
            logger.info(f"Отправка запроса к модели {model_key}")
            
            completion = self.clients[model_key].chat.completions.create(
                model=model_config["id"],
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                extra_headers={
                    "HTTP-Referer": "http://localhost:5002",
                    "X-Title": "AI Chat Assistant",
                }
            )
            
            response = completion.choices[0].message.content
            
            # Сохраняем в контекст
            with self.context_lock:
                context[model_key].append({"role": "user", "content": message})
                context[model_key].append({"role": "assistant", "content": response})
                
                # Ограничиваем размер контекста (оставляем системное сообщение + последние 8 сообщений)
                if len(context[model_key]) > 10:
                    system_msg = context[model_key][0] if context[model_key][0]["role"] == "system" else None
                    context[model_key] = context[model_key][-8:]
                    if system_msg:
                        context[model_key].insert(0, system_msg)
            
            logger.info(f"Получен ответ от модели {model_key}")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка API для модели {model_key}: {e}")
            raise Exception(f"Ошибка API для модели {model_key}: {str(e)}")
    
    def process_with_all_models(self, message: str, session_id: str) -> Dict[str, str]:
        """Обработка сообщения всеми моделями по этапам"""
        results = {}
        
        try:
            logger.info(f"Начало обработки сообщения: {message[:50]}...")
            
            # Этап 1: Генерация LaTeX
            logger.info("Этап 1: Генерация LaTeX")
            latex_prompt = f"Создай LaTeX код для решения задачи: {message}"
            results["latex"] = self.chat_with_model("qwen", latex_prompt, session_id)
            
            # Этап 2: Вычисления (Mistral)
            logger.info("Этап 2: Вычисления")
            calc_prompt = f"Выполни необходимые вычисления для задачи: {message}"
            results["calculations"] = self.chat_with_model("mistral", calc_prompt, session_id)
            
            # Финальная сводка
            results["summary"] = {
                "timestamp": datetime.now().isoformat(),
                "models_used": list(self.models.keys()),
                "session_id": session_id
            }
            
            logger.info("Обработка завершена успешно")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при обработке моделями: {e}")
            logger.error(traceback.format_exc())
            raise Exception(f"Ошибка при обработке: {str(e)}")
    
    def clear_context(self, session_id: str):
        """Очистка контекста сессии"""
        with self.context_lock:
            if session_id in self.conversation_contexts:
                del self.conversation_contexts[session_id]
                logger.info(f"Контекст сессии {session_id} очищен")

# Создаем глобальный экземпляр клиента
ai_client = OpenRouterClient()

@app.route('/')
def serve_html():
    """Главная страница"""
    try:
        return send_from_directory('static', 'index.html')
    except Exception as e:
        logger.error(f"Ошибка при отдаче HTML: {e}")
        return jsonify({"error": "HTML файл не найден"}), 404

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Эндпоинт для обработки чат-сообщений"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Нет данных в запросе"}), 400
        
        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        image = data.get('image')  # Получаем изображение если оно есть
        
        if not message and not image:
            return jsonify({"error": "Необходимо предоставить сообщение или изображение"}), 400
        
        logger.info(f"Получен запрос от сессии {session_id}")
        
        # Если есть изображение, добавляем его в контекст сообщения
        if image:
            message = f"{message}\n[Изображение: {image[:100]}...]" if message else "[Анализ изображения]"
        
        # Обработка всеми моделями
        results = ai_client.process_with_all_models(message, session_id)
        
        return jsonify({
            "response": results,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка в chat_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/chat/single', methods=['POST'])
def single_model_chat():
    """Эндпоинт для общения с одной моделью"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Нет данных в запросе"}), 400
        
        message = data.get('message', '').strip()
        model_key = data.get('model', 'llama_instruct')
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({"error": "Сообщение не может быть пустым"}), 400
        
        if model_key not in ai_client.models:
            return jsonify({"error": f"Неизвестная модель: {model_key}"}), 400
        
        logger.info(f"Запрос к модели {model_key} от сессии {session_id}")
        
        response = ai_client.chat_with_model(model_key, message, session_id)
        
        return jsonify({
            "response": response,
            "model": model_key,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка в single_model_chat: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Эндпоинт для получения списка доступных моделей"""
    try:
        models_info = []
        for key, config in ai_client.models.items():
            models_info.append({
                "key": key,
                "id": config["id"],
                "role": config["role"],
                "description": config["description"]
            })
        
        return jsonify({
            "models": models_info,
            "total": len(models_info),
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Ошибка в get_models: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/context/clear', methods=['POST'])
def clear_context():
    """Очистка контекста сессии"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default') if data else 'default'
        
        ai_client.clear_context(session_id)
        
        return jsonify({
            "message": f"Контекст сессии {session_id} очищен",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Ошибка в clear_context: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния сервера"""
    return jsonify({
        "status": "healthy",
        "message": "AI Chat Server is running",
        "models_available": len(ai_client.models),
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 ошибка: {request.url}")
    return jsonify({"error": "Эндпоинт не найден", "status": "error"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 ошибка: {error}")
    return jsonify({"error": "Внутренняя ошибка сервера", "status": "error"}), 500

def create_static_folder():
    """Создание папки static и копирование HTML файла"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(current_dir, 'static')
    
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        logger.info("Создана папка static")
    
    # Копируем HTML файл в папку static
    source_html = os.path.join(current_dir, 'ai_chat_website.html')
    dest_html = os.path.join(static_dir, 'index.html')
    
    if os.path.exists(source_html):
        import shutil
        shutil.copy2(source_html, dest_html)
        logger.info("HTML файл скопирован в static folder")
    else:
        logger.error("HTML файл не найден")

if __name__ == '__main__':
    create_static_folder()
    
    print("🚀 Запуск AI Chat Server...")
    print("📡 API endpoints:")
    print("   POST /api/chat - обработка всеми моделями")
    print("   POST /api/chat/single - общение с одной моделью")  
    print("   GET /api/models - список моделей")
    print("   POST /api/context/clear - очистка контекста")
    print("   GET /health - проверка состояния")
    print(f"🌐 Server running on http://0.0.0.0:5002")
    
    try:
        app.run(
            host='0.0.0.0',  # Слушаем все интерфейсы
            port=5002,
            debug=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Сервер остановлен")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"❌ Критическая ошибка: {e}")