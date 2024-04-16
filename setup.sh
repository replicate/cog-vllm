#!/bin/bash
set -e
# Function to display help
show_help() {
cat << EOF
Usage: ${0##*/} [-h] [-m MODEL_ID] [-w COG_WEIGHTS] [-t TAG_NAME]

This script configures the environment for COG VLLM by optionally updating
MODEL_ID and COG_WEIGHTS in the .env file, installing COG, and allowing
a custom tag name for the COG build.

    -h          display this help and exit
    -m MODEL_ID specify the model ID to use (e.g., mistralai/Mistral-7B-Instruct-v0.2)
    -w COG_WEIGHTS specify the URL for the COG weights (e.g., https://weights.replicate.delivery/default/mistral-7b-instruct-v0.2)
    -t TAG_NAME  specify the tag name for the COG build (e.g., my-custom-tag)
EOF
}

# Initialize our own variables:
model_id=""
cog_weights=""
tag_name="cog-vllm" # Default tag name

# Process command line arguments
OPTIND=1
while getopts "hm:w:t:" opt; do
    case "$opt" in
    h)
        show_help
        exit 0
        ;;
    m)  model_id=$OPTARG
        ;;
    w)  cog_weights=$OPTARG
        ;;
    t)  tag_name=$OPTARG
        ;;
    esac
done
shift "$((OPTIND-1))" # Shift off the options and optional --.

# Check and replace MODEL_ID in .env if provided
if [ ! -z "$model_id" ]; then
    sed -i "s|^MODEL_ID=.*|MODEL_ID=$model_id|" .env
fi

# Check and replace COG_WEIGHTS in .env if provided
if [ ! -z "$cog_weights" ]; then
    sed -i "s|^COG_WEIGHTS=.*|COG_WEIGHTS=\"$cog_weights\"|" .env
fi

# Install COG
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/download/v0.10.0-alpha5/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog

# Build with COG using the specified tag name
cog build -t $tag_name