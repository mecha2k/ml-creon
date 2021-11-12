from slacker import Slacker
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
bot_user_token = os.getenv("BOT_USER_TOKEN")

slack = Slacker(bot_user_token)
slack.chat.post_message("#stock", "Hello mecha2k (.env) slackers!")
