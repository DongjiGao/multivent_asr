# Introduction
This project aims to automatically segment and transcribe the MultiVENT dataset, reducing the time and effort required for human annotators to make further corrections.
## Description of the recipe
The recipe can be run simply by
```
./run.sh
```
Here is the detailed explanation of the recipe:
### Preparation
This step generates [Lhotse](https://github.com/lhotse-speech/lhotse) CutSets of the MultiVENT dataset for fine-tuning an ASR model. It first converts the video to audio in WAV format with a sampling rate of 16 khz. It then decodes and segments the audio using [WHISPER-timestamped](https://github.com/linto-ai/whisper-timestamped) to produce a CTM file. For segments that are too long (e.g., over 20 seconds), it performs resegmentation to break them down into smaller segments using an LLM (GPT). The resulting data is then processed to extract fbank features and stored as Lhotse CutSets.
```
./prepare.sh \
  --corpus-dir "${corpus_dir}" \
  --lang-dir "${pretrained_lang_dir}"
```
Note: To use GPT for resegmentation, please set the OPENAI_API_KEY by
```
export OPEN_AI_KEY=YOUR_OPEN_AI_KEY
```
### Fine-tuning the Icefall ASR model
We fine-tune a pre-trained [Icefall](https://github.com/k2-fsa/icefall) Zipformer Stateless Transducer model using the data prepared in the previous step.
```
./pruned_transducer_stateless7/finetune.py \
  --world-size ${gpus} \
  --num-epochs 20 \
  --start-epoch 1 \
  --exp-dir "${finetune_out_exp_dir}" \
  --base-lr 0.005 \
  --lr-epochs 100 \
  --lr-batches 100000 \
  --bpe-model "${pretrained_model_dir}/data/lang_bpe_500/bpe.model" \
  --do-finetune True \
  --finetune-ckpt "${pretrained_model_dir}/exp/pretrained.pt" \
  --max-duration 500 \
  --language "en"
```
We tried two pretrained models: one trained on LibriSpeech (1,000 hours) and the other on GigaSpeech (10,000 hours). Here are the fine-tuning results on the test set of MultiVENT created by Dongji Gao:
<table>
  <tr>
    <td></td>
    <td>LibriSpeech</td>
    <td>GigaSpeech</td>
  </tr>
  <tr>
    <td>WER(%)</td> 
    <td>25.06</td>
    <td>36.94</td>
  </tr>
</table>
### OTC flexible alignment

### Post-processing
