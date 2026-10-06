get_converted_path() {
  # strip final slash if exists
  CHECKPOINT_PATH="${1%/}"

  # Split the path into parts using '/'
  IFS='/' read -ra parts <<< "$CHECKPOINT_PATH"

  # Get the number of parts
  local len=${#parts[@]}

  # Get the second-to-last part as RUN_NAME
  local RUN_NAME="${parts[$((len - 2))]}"

  # Get the last part as STEP_STR
  local STEP_STR="${parts[$((len - 1))]}"

  # Construct the OUTPUT_PATH
  local OUTPUT_PATH="/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints/$RUN_NAME/$STEP_STR"

  # Return or echo the output path
  echo "$OUTPUT_PATH"
}
