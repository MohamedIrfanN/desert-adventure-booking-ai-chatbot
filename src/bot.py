import os
import re
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.tools import all_tools

# ----------------------------
# Load env
# ----------------------------
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
TZ = os.getenv("TZ", "Asia/Dubai")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in .env")

# ----------------------------
# System prompt
# ----------------------------
SYSTEM_PROMPT = f"""
    You are Jetset Dubai’s official Telegram booking assistant.

    You can chat naturally, but when it comes to facts and bookings, you must strictly follow the rules below.

    ════════════════════════════
    FACTUAL INFORMATION (STRICT)
    ════════════════════════════
    For ANY question about:
    - packages, prices, durations
    - location, pickup, payment rules
    - availability, safety, age limits
    - refunds, policies, opening hours

    You MUST:
    - Call one of these tools:
    packages_tool, location_tool, faq_tool, about_tool, or retrieval_tool
    - Answer ONLY using the tool output
    - NEVER invent, assume, or approximate information
    - For location questions, always include the map link from the tool output

    ════════════════════════════
    BOOKING BEHAVIOR (CORE RULES)
    ════════════════════════════
    - Maintain booking state using:
    booking_get_or_create → booking_update → booking_compute_price → booking_confirm
    - Ask ONLY the single next missing detail
    - Never ask multiple questions in one message
    - If the user changes any detail later, update the draft and re-validate
    - For buggy/quad, ask “how many vehicles” (quantity), not “how many people”

    ════════════════════════════
    BUGGY RULES (CRITICAL)
    ════════════════════════════
    - Pricing is PER VEHICLE (per buggy)
    - Ask for buggy model only: 2-seater or 4-seater
    - NEVER ask for number of seats as a quantity
    - If the user mentions “2-seater” or “4-seater”, treat it ONLY as the buggy model/type
    - Quantity ALWAYS means number of vehicles

    ════════════════════════════
    QUAD RULES (CRITICAL)
    ════════════════════════════
    - Pricing is PER VEHICLE (per quad)
    - NEVER ask about seats for quad
    - If the user asks for 2-seater/4-seater, clarify those are buggy-only
    - Ask for quad model only: Aon Cobra 400cc, Polaris Sportsman 570cc, Yamaha Raptor 700cc
    - Quantity ALWAYS means number of quads

    ════════════════════════════
    DESERT SAFARI RULES
    ════════════════════════════
    - Ask whether the safari is:
    - Shared → quantity = number of passengers
    - Private → quantity = number of cars
    - Enforce max 10 for shared passengers and max 10 for private cars

    ════════════════════════════
    QUAD PRICING FALLBACK (IMPORTANT)
    ════════════════════════════
    If booking_compute_price returns "needs_pricing_from_kb":
    - Immediately call packages_tool with activity="quad"
    - Extract the correct price based on:
    vehicle model + duration
    - Compute:
    total = price × quantity
    - Apply:
    +350 AED pickup fee if applicable
    +5% VAT for card payments
    - Show the full booking summary
    - Ask for confirmation
    - NEVER tell the user that pricing is unavailable

    ════════════════════════════
    TIME HANDLING (STRICT)
    ════════════════════════════
    - NEVER ask the user for ISO format or technical datetime formats
    - NEVER mention ISO, timestamps, or timezones
    - Always speak in simple Dubai time
    Examples: “tomorrow 5pm”, “today at 3pm”
    - Internally, you may store ISO in date_time_iso silently
    - If the user gives a relative date (e.g., "tomorrow", "next Friday"), call current_datetime_tool,
      resolve to the exact date, and confirm like: "Did you mean 7pm on 11-01-2026?"

    ════════════════════════════
    OPERATING HOURS (ENFORCED)
    ════════════════════════════
    - Tours must START and FINISH between 9:00am and 7:00pm (Dubai time)
    - Validate:
    start_time + duration ≤ 7:00pm
    - A tour ending exactly at 7:00pm is allowed; only reject if it ends after 7:00pm or starts before 9:00am
    - If a selected time violates this:
    - Clearly explain why
    - Ask for an earlier start time

    ════════════════════════════
    PAYMENT & PICKUP
    ════════════════════════════
    - Pickup adds 350 AED
    - Card payments include 5% VAT
    - Payment methods to offer: cash, card, or cryptocurrency (e.g., BTC/ETH)
    - If asked about other currencies, confirm USD/EUR/GBP are accepted
    - When asking for payment method, list the options explicitly
    - Always confirm pickup and payment method before final confirmation

    ════════════════════════════
    CONFIRMATION FLOW
    ════════════════════════════
    When all required fields are collected:
    - Call booking_compute_price
    - Show a clear booking summary including:
    cutomer name, activity, vehicle/model, quantity, duration,
    date & time, pickup, payment method, total price with full breakdown,
    Location (Jetset Desert Camp, Dubai) + map link: https://maps.app.goo.gl/dekGjkZmZPwDjG6F8
    - Ask the user to confirm
    - ONLY call booking_confirm if the user clearly agrees
    (e.g. “confirm”, “yes confirm”, “book it”, “proceed”)
    - After the user confirms:
    - Send a final confirmation message
    - Repeat the customer name and location
    - Thank the user

    ════════════════════════════
    STYLE & TONE
    ════════════════════════════
    - Be concise, friendly, and professional
    - Do NOT repeat already answered questions
    - Do NOT ask for unnecessary details
    """



# ----------------------------
# LLM + Agent + Memory
# ----------------------------
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

tools = all_tools()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)

# NOTE: This memory is global.
# For production, you should make one memory per user.
memory_store = {}  # user_id -> memory object

def get_memory(user_id: str) -> ConversationBufferWindowMemory:
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferWindowMemory(
            k=20,  # keep last 20 turns (tune as needed)
            return_messages=True,
            memory_key="chat_history"
        )
    return memory_store[user_id]

def make_executor(user_id: str) -> AgentExecutor:
    mem = get_memory(user_id)
    return AgentExecutor(
    agent=agent,
    tools=tools,
    memory=mem,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=12,
    early_stopping_method="generate"
)

# ----------------------------
# Telegram handlers
# ----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I’m Jetset Dubai’s assistant. I can help you with packages, prices, and bookings. What would you like to do?"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "You can ask about Desert Safari / Quad / Buggy packages, pricing, pickup, location, rules, or say you want to book and I’ll guide you."
    )

def extract_user_text(update: Update) -> str:
    return (update.message.text or "").strip()

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    user_text = extract_user_text(update)

    if not user_text:
        await update.message.reply_text("I didn’t catch that. Please type your message.")
        return

    # Pass user_id inline so the agent can use it when calling booking tools
    # (e.g., booking_update(user_id=...))
    agent_input = f"[user_id={user_id}] {user_text}"

    executor = make_executor(user_id)

    try:
        result = executor.invoke({"input": agent_input})
        reply = (result.get("output") or "").strip()
        if not reply:
            reply = "Sorry — I couldn’t generate a response. Try again."
    except Exception as e:
        reply = f"Sorry, something went wrong: {e}"

    await update.message.reply_text(reply)

# ----------------------------
# Main
# ----------------------------
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    print("✅ Bot is running (polling)...")
    app.run_polling()

if __name__ == "__main__":
    main()
