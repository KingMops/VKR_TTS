import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, ContextTypes, filters


from scipy.io.wavfile import write as write_wav
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_text_semantic, generate_coarse, generate_fine


TOKEN = '8061300859:AAFGj2zUQ-xReMvQsLworNBX27AU7ReMsQ0'

# Параметры для голоса Jaina
JAINA_SEMANTIC_PATH = 'semantic_output/pytorch_model.bin'
JAINA_COARSE_PATH   = 'coarse_output/pytorch_model.bin'
JAINA_FINE_PATH     = 'fine_output/pytorch_model.bin'
JAINA_VOICE_NAME    = 'datasets/jaina/tokens/jaina_1.npz'
JAINA_SEMANTIC_TEMP = 0.7 #было 0,6
JAINA_SEMANTIC_TOP_K = 10 #было 40
JAINA_SEMANTIC_TOP_P = 0.9 #было 0,85
JAINA_COARSE_TEMP   = 0.3
JAINA_COARSE_TOP_K  = 10 #было 50
JAINA_COARSE_TOP_P  = 0.9 #было 0,95
JAINA_FINE_TEMP     = 0.5 #было 0,4
JAINA_USE_SEMANTIC_HISTORY_PROMPT = False
JAINA_USE_COARSE_HISTORY_PROMPT   = True
JAINA_USE_FINE_HISTORY_PROMPT     = True
JAINA_OUTPUT_FULL   = False

# Параметры для голоса Yennefer
YENNEFER_SEMANTIC_PATH = 'semantic_output_yennefer/pytorch_model.bin'
YENNEFER_COARSE_PATH   = 'coarse_output_yennefer/pytorch_model.bin'
YENNEFER_FINE_PATH     = 'fine_output_yennefer/pytorch_model.bin'
YENNEFER_VOICE_NAME    = 'datasets/yennefer/tokens/yennefer_8.npz'
YENNEFER_SEMANTIC_TEMP = 0.7
YENNEFER_SEMANTIC_TOP_K = 50
YENNEFER_SEMANTIC_TOP_P = 0.9
YENNEFER_COARSE_TEMP   = 0.3
YENNEFER_COARSE_TOP_K  = 50
YENNEFER_COARSE_TOP_P  = 0.9 #было 0,95
YENNEFER_FINE_TEMP     = 0.5
YENNEFER_USE_SEMANTIC_HISTORY_PROMPT = True
YENNEFER_USE_COARSE_HISTORY_PROMPT   = True
YENNEFER_USE_FINE_HISTORY_PROMPT     = True
YENNEFER_OUTPUT_FULL   = False


def generate_with_voice_config(text_prompt, cfg: dict):

    preload_models(
        text_use_gpu=True,
        text_use_small=False,
        text_model_path=cfg['semantic_path'],
        coarse_model_path=cfg['coarse_path'],
        fine_model_path=cfg['fine_path'],
        coarse_use_gpu=True,
        fine_use_gpu=True,
        codec_use_gpu=True,
        force_reload=True, 
        path="models_temp", 
    )

    x_sem = generate_text_semantic(
        text_prompt,
        history_prompt=cfg['voice_name'] if cfg.get('use_semantic_history_prompt') else None,
        temp=cfg['semantic_temp'],
        top_k=cfg['semantic_top_k'],
        top_p=cfg['semantic_top_p'],
    )

    x_coarse = generate_coarse(
        x_sem,
        history_prompt=cfg['voice_name'] if cfg.get('use_coarse_history_prompt') else None,
        temp=cfg['coarse_temp'],
        top_k=cfg['coarse_top_k'],
        top_p=cfg['coarse_top_p'],
    )

    x_fine = generate_fine(
        x_coarse,
        history_prompt=cfg['voice_name'] if cfg.get('use_fine_history_prompt') else None,
        temp=cfg['fine_temp'],
    )
    return codec_decode(x_fine)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Jaina", callback_data='select_jaina')],
        [InlineKeyboardButton("Yennefer", callback_data='select_yennefer')],
    ]
    await update.message.reply_text(
        'Привет! Выбери модель голоса:',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )




async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    key = query.data.replace('select_', '')
    if key in ['jaina', 'yennefer']:
        context.user_data['voice_key'] = key
        context.user_data['awaiting_text'] = True
        await query.message.reply_text(
            f'Выбран голос «{key}». Теперь отправь мне текст для генерации аудио.'
        )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get('awaiting_text'):
        return

    text = update.message.text
    voice_key = context.user_data.get('voice_key', 'jaina')


    if voice_key == 'jaina':
        cfg = {
            'semantic_path': JAINA_SEMANTIC_PATH,
            'coarse_path': JAINA_COARSE_PATH,
            'fine_path': JAINA_FINE_PATH,
            'voice_name': JAINA_VOICE_NAME,
            'semantic_temp': JAINA_SEMANTIC_TEMP,
            'semantic_top_k': JAINA_SEMANTIC_TOP_K,
            'semantic_top_p': JAINA_SEMANTIC_TOP_P,
            'coarse_temp': JAINA_COARSE_TEMP,
            'coarse_top_k': JAINA_COARSE_TOP_K,
            'coarse_top_p': JAINA_COARSE_TOP_P,
            'fine_temp': JAINA_FINE_TEMP,
            'use_semantic_history_prompt': JAINA_USE_SEMANTIC_HISTORY_PROMPT,
            'use_coarse_history_prompt': JAINA_USE_COARSE_HISTORY_PROMPT,
            'use_fine_history_prompt': JAINA_USE_FINE_HISTORY_PROMPT,
            'output_full': JAINA_OUTPUT_FULL,
        }
    elif voice_key == 'yennefer':
        cfg = {
            'semantic_path': YENNEFER_SEMANTIC_PATH,
            'coarse_path': YENNEFER_COARSE_PATH,
            'fine_path': YENNEFER_FINE_PATH,
            'voice_name': YENNEFER_VOICE_NAME,
            'semantic_temp': YENNEFER_SEMANTIC_TEMP,
            'semantic_top_k': YENNEFER_SEMANTIC_TOP_K,
            'semantic_top_p': YENNEFER_SEMANTIC_TOP_P,
            'coarse_temp': YENNEFER_COARSE_TEMP,
            'coarse_top_k': YENNEFER_COARSE_TOP_K,
            'coarse_top_p': YENNEFER_COARSE_TOP_P,
            'fine_temp': YENNEFER_FINE_TEMP,
            'use_semantic_history_prompt': YENNEFER_USE_SEMANTIC_HISTORY_PROMPT,
            'use_coarse_history_prompt': YENNEFER_USE_COARSE_HISTORY_PROMPT,
            'use_fine_history_prompt': YENNEFER_USE_FINE_HISTORY_PROMPT,
            'output_full': YENNEFER_OUTPUT_FULL,
        }


    await update.message.reply_text('Генерирую аудио… Пожалуйста, подожди несколько секунд.')
    audio = generate_with_voice_config(text, cfg)

    os.makedirs('audio', exist_ok=True)
    path = os.path.join('audio', f'{update.effective_user.id}_{voice_key}.wav')
    write_wav(path, SAMPLE_RATE, audio)

    with open(path, 'rb') as f:
        await context.bot.send_audio(
            chat_id=update.effective_chat.id,
            audio=f,
            title=f"Аудио ({voice_key})"
        )

    context.user_data.clear()


def main() -> None:
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    print("Бот запущен.")
    app.run_polling()


if __name__ == '__main__':
    main()
