from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from io import BytesIO
import discord
import asyncio
import os
import numpy as np
from PIL import Image

from dotenv import load_dotenv

load_dotenv()

# MODEL_ID = "stabilityai/stable-diffusion-2-base"
# MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_ID = "stabilityai/stable-diffusion-2-1"

ALT_MODEL_ID = "prompthero/openjourney-v2"
# UPSCALER_ID = "caidas/swin2SR-classical-sr-x2-64"

PIPE_DRIVER = os.getenv("PIPE_DRIVER")

class AiService:
    def __init__(self):
        print("Loading AI Models...")
        print("Loading Primary")
        self.scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, scheduler=self.scheduler, revision="fp16", torch_dtype=torch.float16)
        self.pipe.to(PIPE_DRIVER)
        self.pipe.enable_attention_slicing()

        print("Loading Alt")
        self.altpipe = StableDiffusionPipeline.from_pretrained(ALT_MODEL_ID, torch_dtype=torch.float16)
        self.altpipe.to(PIPE_DRIVER)
        # self.altpipe.scheduler = DPMSolverMultistepScheduler.from_config(self.altpipe.scheduler.config)
        self.altpipe.enable_attention_slicing()

        self._queue = asyncio.Queue()

        # print("Loading Upscaler")
        # self.upscaler = Swin2SRForImageSuperResolution.from_pretrained(UPSCALER_ID)
        # self.upscaler.to("cuda")
        # self.upscaler.enable_attention_slicing()

        # print("Loading Image Processor")
        # self.upscale_processor = AutoImageProcessor.from_pretrained(UPSCALER_ID)

    async def queue(self, ctx, prompt, negative="", upscale=False, alt=False):
        print(f"Queuing {prompt}...")

        if negative:
            await ctx.respond(f"Generating `{prompt}` without `{negative}`...")
        else:
            await ctx.respond(f"Generating `{prompt}`...")

        await self._queue.put((ctx, prompt, negative, upscale, alt))
        if not hasattr(self, 'worker_task'):
            self.worker_task = asyncio.create_task(self.worker())

    async def worker(self):
        while True:
            ctx, prompt, negative, upscale, alt = await self._queue.get()
            if alt:
                print(f"Alt - Generating {prompt}...")
                image = self.altpipe(prompt=prompt, negative_prompt=negative).images[0]
            else:
                print(f"Generating {prompt}...")
                image = self.pipe(prompt=prompt, negative_prompt=negative).images[0]

            if upscale:
                await ctx.edit(f"Upscaling no worky for {prompt}...")
                # continue

                # print(f"Upscaling {prompt}...")
                # await ctx.respond(f"Upscaling {prompt}...")
                # inputs = self.upscale_processor(image, return_tensors="pt")
                # with torch.no_grad():
                #     outputs = self.upscaler(**inputs)

                #     output = outputs.reconstruction.data.squeeze().float().cuda().clamp_(0, 1).numpy()
                #     output = np.moveaxis(output, source=0, destination=-1)
                #     output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
                #     image = Image.fromarray(output)
                #     image.show()

                # image = self.upscaler(prompt=prompt, image=image).images[0]

            with BytesIO() as image_binary:
                image.save(image_binary, 'PNG')
                image_binary.seek(0)

                picture = discord.File(fp = image_binary, filename='image.png')

                await ctx.edit(file=picture)
            print(f"Completed {prompt}")
            self._queue.task_done()
