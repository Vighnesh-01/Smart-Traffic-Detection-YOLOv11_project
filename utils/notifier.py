import threading
import requests
import logging

logger = logging.getLogger(__name__)

def _send_logic(image_path, message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(image_path, 'rb') as photo:
            payload = {'chat_id': chat_id, 'caption': message}
            files = {'photo': photo}
            requests.post(url, data=payload, files=files, timeout=10)
    except Exception as e:
        logger.error(f"Background Telegram error: {e}")

def send_telegram_alert(image_path, message, token, chat_id):
    # This creates a 'fire and forget' thread so your video stays smooth
    threading.Thread(
        target=_send_logic, 
        args=(image_path, message, token, chat_id), 
        daemon=True
    ).start()