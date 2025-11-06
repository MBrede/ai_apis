import json
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext, MessageHandler, filters
import dotenv
import os
import datetime
import requests
from io import BytesIO
import tempfile

dotenv.load_dotenv('.env')
API_ENDPOINT = "http://149.222.209.100:1234"
BOT_TOKEN = os.environ["telegram_token"]
try:
    with open('users.json', 'r') as f:
        USERS = json.load(f)
except FileNotFoundError:
    USERS = {}

try:
    with open('contacts.json', 'r') as f:
        CONTACTS = json.load(f)
except FileNotFoundError:
    CONTACTS = {}


def save_users():
    with open('users.json', 'w') as f:
        json.dump(USERS, f, indent=4)


def update_contact_attempts(user_id):
    global CONTACTS
    CONTACTS[user_id] = str(datetime.datetime.now())
    with open('contacts.json', 'w') as f:
        json.dump(CONTACTS, f, indent=4)


base_settings = {
    "model_id": "runwayml/stable-diffusion-v1-5",
    "torch_dtype": "float16",
    "num_inference_steps": 20,
    "count_returned": 1,
    "seed": 0,
    "guidance_scale": 7.5,
    "negative_prompt": "blurry, low resolution, low quality",
    "width": 512,
    "height": 512,
    "lora": ''
}


async def check_privileges(update, admin_function=False):
    user_id = str(update.effective_user.id)
    if user_id not in USERS:
        update_contact_attempts(user_id)
        await update.message.reply_text(f"You are not on the list, user {user_id}!"
                                        f"Now sod off!")
        return 0
    elif USERS[user_id]["admin"]:
        return 2
    elif admin_function and not USERS[user_id]["admin"]:
        return 0
    else:
        return 1


async def start(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update)
    if privileges == 2:
        await update.message.reply_text("Hello Admin! Type /help for commands.")
    elif privileges:
        await update.message.reply_text("Welcome! Please type /help to see available commands.")


async def help_command(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update)
    if privileges:
        help_text = ""
        if privileges == 2:
            help_text = ("Admin Commands:\n"
                         "/add_admin <user_id> - Grant admin status to a user\n"
                         "/remove_admin <user_id> - Revoke admin status from a user\n"
                         "/add_lora <lora_name> - Add a new LORA model\n"
                         "/add_user <user_id> - Add a user to admin list\n"
                         "/del_user <user_id> - Remove a user from admin list\n"
                         "/list_users - List all admins\n"
                         "/list_contacts - List all recent contact attempts\n"
                         "\n"
                         )
        help_text += ("User Commands:\n"
                      "/get_parameters - Get current parameter-settings\n"
                      "/get_loras - Get available LORAs\n"
                      "/get_sd - Get available stable diffusions\n"
                      "/set_parameters - Set one or more model parameters\n"
                      "/llm - switch between sd and llm-mode\n"
                      "/assist <text> - Get help creating prompts with the LLM\n"
                      "/assist_create <text> - Get help creating prompts and let SD directly turn it into an image\n"
                      "/img2prompt <image> - Describe an image."
                      "Just write Text - Will be interpreted as a prompt for image generation"
                      )
        await update.message.reply_text(help_text)


# Adjusting message_handler to support text and image
async def message_handler(update: Update, context: CallbackContext) -> None:
    user_id = str(update.effective_user.id)
    privileges = await check_privileges(update)
    if privileges:
        current_settings = USERS[user_id]['current_settings']
        if update.message.photo:
            await update.message.reply_text("You sent an image")
            photo = update.message.photo[-1]  # Get the highest resolution photo
            prompt = update.message.caption
            if prompt != '/img2prompt':
                await handle_photo_prompt(photo, prompt, update, context, current_settings)
            else:
                await img2prompt_handler(update, context, photo)
        elif update.message.audio:
            sound = update.message.audio
            await audio_transcription(update, context, sound)
        elif update.message.voice:
            sound = update.message.voice
            await audio_transcription(update, context, sound)
        else:
            if USERS[user_id]['mode'] == 'llm':
                await llm_handler(update, context)
            else:
                prompt = update.message.text
                await handle_text_prompt(prompt, update, context, current_settings)


# Handler for text prompts
async def handle_text_prompt(prompt, update, context, settings):
    await update.message.reply_text("Generating image based on the prompt:\n" + prompt)
    url = f"{API_ENDPOINT}/post_config?" + '&'.join([f'{k}={settings[k]}' for k in settings]) + f"&prompt={prompt}"
    response = requests.post(url)
    if response.status_code == 200:
        image_stream = BytesIO(response.content)
        image_stream.seek(0)
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_stream)
    else:
        await update.message.reply_text("Failed to generate image.")


# Handler for photo prompts
async def handle_photo_prompt(photo, prompt, update, context, settings):
    photo_file = await context.bot.get_file(photo.file_id)
    photo_bytes = await photo_file.download_as_bytearray()
    files = {'image': ('image.jpg', photo_bytes, 'multipart/form-data', {'Expires': '0'})}
    url = f"{API_ENDPOINT}/post_config?" + f"prompt={prompt}&" + '&'.join([f'{k}={settings[k]}' for k in settings])
    response = requests.post(url, files=files)
    if response.status_code == 200:
        image_stream = BytesIO(response.content)
        image_stream.seek(0)
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_stream)
    else:
        await update.message.reply_text("Failed to process the image.")


async def add_admin(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update, admin_function=True)
    if privileges == 2 and context.args:
        new_admin_id = context.args[0]
        if new_admin_id in USERS:
            USERS[new_admin_id]["admin"] = True
            save_users()
            await update.message.reply_text(f"User {new_admin_id} has been granted admin status.")
        else:
            await update.message.reply_text("User ID not found.")
    else:
        await update.message.reply_text("You do not have permission to perform this action.")


async def remove_admin(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update, admin_function=True)
    if privileges == 2 and context.args:
        admin_id = context.args[0]
        if admin_id in USERS and USERS[admin_id]["admin"]:
            USERS[admin_id]["admin"] = False
            save_users()
            await update.message.reply_text(f"Admin status removed from user {admin_id}.")
        else:
            await update.message.reply_text("User ID not found or user is not an admin.")


async def change_mode(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update, admin_function=False)
    if privileges >= 1:
        user_id = str(update.effective_user.id)
        if user_id in USERS:
            if USERS[user_id]["mode"] == "sd":
                USERS[user_id]["mode"] = "llm"
            else:
                USERS[user_id]["mode"] = "sd"
            save_users()
        await update.message.reply_text(f"You are now in {USERS[user_id]['mode']} mode.")

async def add_user(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update, admin_function=True)
    if privileges == 2 and context.args:
        new_user_id = context.args[0]
        USERS[new_user_id] = {"admin": False,
                              "mode": "sd",
                              "current_settings": base_settings}  # default new users as non-admin
        save_users()
        await update.message.reply_text(f"User {new_user_id} added to the system.")
    else:
        await update.message.reply_text("You do not have permission to perform this action.")


async def del_user(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update, admin_function=True)
    if privileges == 2 and context.args:
        user_id = context.args[0]
        if user_id in USERS:
            del USERS[user_id]
            save_users()
            await update.message.reply_text(f"User {user_id} removed from the system.")
        else:
            await update.message.reply_text("User ID not found.")
    else:
        await update.message.reply_text("You do not have permission to perform this action.")


async def list_users(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update, admin_function=True)
    if privileges == 2:
        user_list = '\n'.join([f"{user_id}: {'Admin' if USERS[user_id]['admin'] else 'User'}" for user_id in USERS])
        await update.message.reply_text(f"List of Users:\n{user_list}")
    else:
        await update.message.reply_text("You do not have permission to view this information.")


async def list_contacts(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update, admin_function=True)
    if privileges == 2:
        contact_list = '\n'.join([f"{user_id}: Last contact on {CONTACTS[user_id]}" for user_id in CONTACTS])
        await update.message.reply_text(f"Recent Contacts:\n{contact_list}")
    else:
        await update.message.reply_text("You do not have permission to view this information.")


async def add_lora(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update, admin_function=True)
    if privileges == 2:
        if context.args:
            lora_name = ' '.join(context.args)
            response = requests.post(f"{API_ENDPOINT}/add_new_lora?name={lora_name}")
            if response.ok:
                await update.message.reply_text(f"LORA model '{lora_name}' added successfully."
                                                f"API responded: {response.content}")
            else:
                await update.message.reply_text("Failed to add LORA model.")
        else:
            await update.message.reply_text("Please add the name of the model after the command.")
    else:
        await update.message.reply_text("You do not have permission to perform this action.")


async def get_parameters(update: Update, context: CallbackContext) -> None:
    """ Displays the current settable parameters for the model. """
    privileges = await check_privileges(update)
    if privileges:
        user_id = str(update.effective_user.id)
        help_text = "Your current settings are:\n"
        for key in USERS[user_id]['current_settings']:
            help_text += f"{key}: {USERS[user_id]['current_settings'][key]}\n"
        await update.message.reply_text(help_text)
    else:
        await update.message.reply_text("You do not have permission to access this information.")


async def get_loras(update: Update, context: CallbackContext) -> None:
    """ Fetches and displays the list of available LORA models from the API. """
    privileges = await check_privileges(update)
    if privileges:
        response = requests.get(f"{API_ENDPOINT}/get_available_loras")
        if response.status_code == 200:
            loras = response.json()
            loras_text = "Available LORAs:\n"
            for lora in loras:
                if 'trigger words' in loras[lora] and loras[lora]['trigger words']:
                    loras_text += f"{lora}:\n trigger words:{', '.join(loras[lora]['trigger words'])}\n\n"
                else:
                    loras_text += f"{lora}\n\n"
            await update.message.reply_text(loras_text)
        else:
            await update.message.reply_text("Failed to fetch LORA models.")


async def get_sd(update: Update, context: CallbackContext) -> None:
    """ Fetches and displays the list of available stable diffusion models. """
    privileges = await check_privileges(update)
    if privileges:
        response = requests.get(f"{API_ENDPOINT}/get_available_stable_diffs")
        if response.status_code == 200:
            sd_models = response.json()
            sd_text = "Available Stable Diffusion Models:\n" + '\n'.join(sd_models.values())
            await update.message.reply_text(sd_text)
        else:
            await update.message.reply_text("Failed to fetch stable diffusion models.")


async def set_parameters(update: Update, context: CallbackContext) -> None:
    """ Sets one or more model parameters dynamically. """
    privileges = await check_privileges(update)
    if privileges:
        global USERS
        user_id = str(update.effective_user.id)
        current_settings = USERS[user_id]['current_settings']
        if not context.args:
            await update.message.reply_text(
                "Please provide parameters in the format: /set_parameters key=value key2=value2")
            return
        params = dict(arg.split('=') for arg in context.args)
        changes = []
        for param, value in params.items():
            if param in current_settings:
                try:
                    # Convert integers and floats from strings
                    if param in ['num_inference_steps', 'count_returned', 'seed', 'width', 'height']:
                        value = int(value)
                    elif param in ['guidance_scale']:
                        value = float(value)
                    current_settings[param] = value
                    changes.append(f"Updated {param} to {value}")
                except ValueError:
                    await update.message.reply_text(f"Invalid value for {param}.")
                    return
        if changes:
            USERS[user_id]['current_settings'] = current_settings
            save_users()
            await update.message.reply_text("Changes made:\n" + '\n'.join(changes))
        else:
            await update.message.reply_text("No valid parameters provided.")


async def llm_handler(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update)
    if privileges:
        text = update.message.text
        if not text:
            await update.message.reply_text("Please provide some text for the LLM interface.")
            return
        response = requests.get(f"{API_ENDPOINT}/llm_interface", params={"text": text})
        if response.status_code == 200:
            generated_text = response.text
            await update.message.reply_text(generated_text)
        else:
            await update.message.reply_text("Failed to generate response from LLM.")


async def assist_handler(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update)
    if privileges:
        text = ' '.join(context.args)
        if not text:
            await update.message.reply_text("Please provide some text for prompt assistance.")
            return
        await update.message.reply_text("Generating answer...")
        response = requests.get(f"{API_ENDPOINT}/llm_prompt_assistance", params={"text": text})
        if response.status_code == 200:
            prompt_description = response.text.split('Description:')[-1][:-1]
            await update.message.reply_text(prompt_description)
        else:
            await update.message.reply_text("Failed to get prompt assistance from LLM.")


async def assist_creator(update: Update, context: CallbackContext) -> None:
    privileges = await check_privileges(update)
    if privileges:
        text = ' '.join(context.args)
        if not text:
            await update.message.reply_text("Please provide some text for prompt assistance.")
            return
        await update.message.reply_text("Generating answer...")
        response = requests.get(f"{API_ENDPOINT}/llm_prompt_assistance", params={"text": text})
        if response.status_code == 200:
            prompt = response.text.split('Description:')[-1][:-1]
            user_id = str(update.effective_user.id)
            current_settings = USERS[user_id]['current_settings']
            await handle_text_prompt(prompt, update, context, current_settings)
        else:
            await update.message.reply_text("Failed to get prompt assistance from LLM.")

async def img2prompt_handler(update: Update, context: CallbackContext, photo) -> None:
    privileges = await check_privileges(update)
    if privileges:
        photo_file = await context.bot.get_file(photo.file_id)
        photo_bytes = await photo_file.download_as_bytearray()
        files = {'image': ('image.jpg', BytesIO(photo_bytes), 'image/jpeg')}
        url = f"{API_ENDPOINT[:-4]}8000/img2prompt"
        response = requests.post(url, files=files)
        if response.status_code == 200:
            prompt_text = response.text
            await update.message.reply_text(f"Generated text: {prompt_text}")
        else:
            await update.message.reply_text("Failed to generate text from image.")

async def audio_transcription(update: Update, context: CallbackContext, sound) -> None:
    privileges = await check_privileges(update)
    if privileges:
        await update.message.reply_text(f"Transcribing {sound.duration}s long audio.")
        if 'audio' in sound.mime_type:
            file = await context.bot.get_file(sound.file_id)
            audio_bytes = await file.download_as_bytearray()
            with tempfile.NamedTemporaryFile(delete=False, 
                                             suffix=sound.mime_type.split('/')[1]) as tmp:
                tmp.write(audio_bytes) 
                files = { 'file': (tmp.name, open(tmp.name, 'rb'), sound.mime_type)}
                url = "http://149.222.209.100:8080/transcribe?model_to_use=turbo"
                response = requests.post(url, files=files)
                
                transcription = json.loads(response.text)['answer']
                
                # Split transcription into chunks if it's too long
                # Telegram message limit is 4096 characters
                MAX_LENGTH = 4000  # Leave some buffer
                
                if len(transcription) <= MAX_LENGTH:
                    await update.message.reply_text(f"Transcription: {transcription}")
                else:
                    # Split by sentences or at word boundaries to avoid cutting words
                    chunks = []
                    current_chunk = "Transcription (part 1): "
                    part_num = 1
                    
                    # Split into sentences (basic approach)
                    sentences = transcription.replace('. ', '.|').replace('? ', '?|').replace('! ', '!|').split('|')
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= MAX_LENGTH:
                            current_chunk += sentence + " "
                        else:
                            chunks.append(current_chunk.strip())
                            part_num += 1
                            current_chunk = f"Transcription (part {part_num}): " + sentence + " "
                    
                    # Add the last chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    
                    # Send all chunks
                    for chunk in chunks:
                        await update.message.reply_text(chunk)
        else:
            await update.message.reply_text("Sound was not in a usable format.")

def main() -> None:
    """Starts the bot and registers all command and message handlers."""
    # Create the Application using the bot token from the environment variables
    app = Application.builder().token(BOT_TOKEN).connect_timeout(60).read_timeout(60).write_timeout(60).build()

    # Registering command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("add_admin", add_admin))
    app.add_handler(CommandHandler("remove_admin", remove_admin))
    app.add_handler(CommandHandler("add_user", add_user))
    app.add_handler(CommandHandler("del_user", del_user))
    app.add_handler(CommandHandler("list_users", list_users))
    app.add_handler(CommandHandler("list_contacts", list_contacts))
    app.add_handler(CommandHandler("add_lora", add_lora))
    app.add_handler(CommandHandler("get_parameters", get_parameters))
    app.add_handler(CommandHandler("get_loras", get_loras))
    app.add_handler(CommandHandler("get_sd", get_sd))
    app.add_handler(CommandHandler("set_parameters", set_parameters))
    app.add_handler(CommandHandler("llm", change_mode))
    app.add_handler(CommandHandler("assist", assist_handler))
    app.add_handler(CommandHandler("assist_create", assist_creator))

    # Registering a message handler for handling generic text input
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE | filters.PHOTO | filters.TEXT & ~filters.COMMAND, message_handler))

    # Start the bot until the user stops it
    app.run_polling()


if __name__ == '__main__':
    main()
