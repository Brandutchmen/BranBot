import discord
from discord.ext import commands
from MemeService import MemeService
from AiService import AiService
import os
from dotenv import load_dotenv
load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
guild_ids = [1054203182294782052]
bot = commands.Bot()
bb = bot.create_group("bb", "BranBot commands")

ai = bot.create_group("ai", "BranBot AI commands")

ais = AiService()


@bb.command(description="Sends the bot's latency.", guild_ids=guild_ids)
async def ping(ctx):
    await ctx.respond(f"Pong! Latency is {bot.latency}")

@bb.command(description="Sends memes", guild_ids=guild_ids)
async def meme(ctx):
    with open(MemeService.get_random_meme(), 'rb') as f:
        picture = discord.File(f)
        await ctx.respond(file=picture)

@ai.command(description="Generate AI Art, BranBot style ;)", guild_ids=guild_ids)
async def gen(ctx, prompt, negative="", upscale=False, alt=False):
    await ais.queue(ctx, prompt, negative, upscale, alt)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')
    print(f'Guild ID: {bot.guilds[0].id}')

def main():
    bot.run(os.getenv('DISCORD_ID'))

if __name__ == '__main__':
    main()
