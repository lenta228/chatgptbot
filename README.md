**Discord ChatGPT Bot**

**Description**
Discord bot using GPT models to respond to user messages in channels and forum threads.

**Features**
• Responds to direct and role mentions
• Formats text for Discord properly
• Shows uptime in status
• Handles forum threads

**Installation**
1. Install dependencies: `pip install discord.py g4f python-dotenv websockets`
2. Create `.env` file with `DISCORD_TOKEN=your_token_here`
3. Run with: `python discordbot.py`

**Usage**
Mention the bot in a Discord channel with your question: `@BotName What is the capital of France?`

**Troubleshooting**
• Check console for error messages
• Verify bot permissions in your server
• Try alternative GPT providers if needed

**Credits**
Uses g4f library for LLM access. 