# Introduction
This project aims to automatically segment and transcribe the MultiVENT dataset, reducing the time and effort required for human annotators to make further corrections.
## Description of the recipe
### Preparation
This step generates [Lhotsep](https://github.com/lhotse-speech/lhotse) CutSets of the MultiVENT dataset for fine-tuning an ASR model. It first converts the video to audio in WAV format with a sampling rate of 16 khz. It then decodes and segments the audio using [WHISPER-timestamped](https://github.com/linto-ai/whisper-timestamped) to produce a CTM file. For segments that are too long (e.g., over 20 seconds), it performs resegmentation to break them down into smaller segments using an LLM (GPT). The resulting data is then processed to extract fbank features and stored as Lhotse CutSets.
```
./prepare.sh \
  --corpus-dir "${corpus_dir}" \
  --lang-dir "${pretrained_lang_dir}"
```
Note: To use GPT for resegmentation, please set the OPENAI_API_KEY by
```
export OPEN_AI_KEY=YOUR_OPEN_AI_KEY
```
