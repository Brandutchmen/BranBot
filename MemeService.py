import os
import random

class MemeService:

    @staticmethod
    def get_random_meme():
        # Get random meme from memes folder
        memes = os.listdir('memes')
        # return random meme
        return f'memes/{memes[random.randint(0, len(memes) - 1)]}'