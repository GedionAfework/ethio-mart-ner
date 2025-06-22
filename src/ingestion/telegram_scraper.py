import os
import csv
import asyncio
from datetime import datetime
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from dotenv import load_dotenv

load_dotenv()
api_id = os.getenv("TG_API_ID")
api_hash = os.getenv("TG_API_HASH")
phone = os.getenv("PHONE")

client = TelegramClient("scraping_session", api_id, api_hash)


async def scrape_channel(client, channel_username, writer, media_dir):
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title
        async for message in client.iter_messages(entity, limit=100):
            media_path = None
            if message.media and hasattr(message.media, "photo"):
                filename = f"{channel_username.lstrip('@')}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                await client.download_media(message.media, media_path)

            views = message.views if message.views else 0
            timestamp = message.date.isoformat()
            message_text = message.message or ""

            writer.writerow(
                [
                    channel_title,
                    channel_username,
                    message.id,
                    message_text,
                    timestamp,
                    views,
                    media_path,
                ]
            )
        print(f"Successfully scraped {channel_username}")
    except Exception as e:
        print(f"Error scraping {channel_username}: {e}")


async def main():
    await client.start(phone)

    raw_data_dir = "data/raw"
    media_dir = "data/raw/photos"
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(media_dir, exist_ok=True)

    output_file = os.path.join(
        raw_data_dir, f'telegram_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

    channels = [
        "@Shageronlinestore",
        "@ZemenExpress",
        "@nevacomputer",
        "@meneshayeofficial",
        "@ethio_brand_collection",
        "@Leyueqa",
        "@sinayelj",
        "@Shewabrand",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Channel Title",
                "Channel Username",
                "Message ID",
                "Message Text",
                "Timestamp",
                "Views",
                "Media Path",
            ]
        )

        for channel in channels:
            await scrape_channel(client, channel, writer, media_dir)

    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(main())
