import threading
import requests
import logging

logger = logging.getLogger(__name__)

def _send_logic(image_path, message, token, chat_id):
    """
    INTERNAL: Handles the actual network request.
    This runs in a background thread to keep the main video smooth.
    """
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    payload = {'chat_id': chat_id, 'caption': message}
    
    try:
        if image_path:
            with open(image_path, 'rb') as photo:
                # Use a short timeout (3s-5s). 
                # If the internet is slow, the thread dies, but the VIDEO stays smooth.
                requests.post(url, data=payload, files={'photo': photo}, timeout=5)
        else:
            # If no photo, send as a regular message
            msg_url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(msg_url, data=payload, timeout=3)
            
    except requests.exceptions.RequestException as e:
        # This catches timeouts, DNS issues, and connection drops
        logger.error(f"Telegram background upload failed: {e}")

def send_telegram_alert(image_path, message, token, chat_id):
    """
    EXTERNAL: Spawns a 'Fire and Forget' thread.
    Use this in your main loop.
    """
    # daemon=True ensures these threads close when you quit the main program
    thread = threading.Thread(
        target=_send_logic, 
        args=(image_path, message, token, chat_id), 
        daemon=True
    )
    thread.start()