#!/bin/sh
# Assumes COG_WEIGHTS is set in predict.py
# COG_WEIGHTS="Qwen/Qwen3-VL-8B-Instruct"
# OR
# COG_WEIGHTS="https://weights.../Qwen3-VL-8B-Instruct/model.tar"

exec cog predict -i prompt="Who are you?"
