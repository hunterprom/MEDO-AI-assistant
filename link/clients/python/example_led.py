"""Minimal medolink example: an LED you can voice-control + a sensor report.

    pip install -e link/clients/python
    python example_led.py 192.168.1.20:8710 <token from secrets.local.yaml>

Then say: "led on", "led off", or ask MEDO "what does the pi sensor read".
"""

import asyncio
import random
import sys

from medolink import MedoLink

MANIFEST = {
    "device_id": "pi_led",
    "name": "the Pi LED",
    "description": "Raspberry Pi demo: one LED and a fake sensor.",
    "capabilities": [
        {"name": "led_on", "description": "Turn the Pi's LED on.",
         "fast_patterns": ["\\bled on\\b"]},
        {"name": "led_off", "description": "Turn the Pi's LED off.",
         "fast_patterns": ["\\bled off\\b"]},
        {"name": "read_sensor",
         "description": "Read the Pi's sensor value and report it."},
    ],
}

link = MedoLink(sys.argv[1], token=sys.argv[2], manifest=MANIFEST)
led = {"on": False}


@link.on("led_on")
async def led_on(params):
    led["on"] = True                    # real hardware: GPIO.output(pin, HIGH)
    return "LED is on."


@link.on("led_off")
async def led_off(params):
    led["on"] = False
    return "LED is off."


@link.on("read_sensor")
async def read_sensor(params):
    return f"The sensor reads {random.randint(18, 26)} degrees."


if __name__ == "__main__":
    asyncio.run(link.run())
