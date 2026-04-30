import asyncio
import asyncpg
import sys

async def test():
    for pw in ['postgres', 'admin', 'root', 'password', '1234', '', '12345']:
        try:
            conn = await asyncpg.connect(user='postgres', password=pw, database='postgres', host='localhost')
            print(f'SUCCESS:{pw}')
            await conn.close()
            return
        except asyncpg.exceptions.InvalidPasswordError:
            pass
        except Exception as e:
            pass
    print("FAILED")

asyncio.run(test())
