# EE 679 Auto-Subtitling Experiment Report

## Setup
- Speech dataset: LibriSpeech `test-clean` subset (8 samples, 1 sample per speaker)
- Noise conditions: white, pink, soundtrack
- SNR levels: 20 dB, 10 dB, 0 dB
- ASR model: faster-whisper `tiny.en`
- Front-end systems: raw, enhanced (Wiener), adaptive (MFCC + spectral rules)

## Main Findings
- Best overall system: **raw**
- Mean WER of best system: **0.231**
- Mean CER of best system: **0.081**
- Subtitle timestamp MAE: **5.000 sec**
- Predicted subtitle CPS violations: **3**

## System Summary
```text
  system  snr_db  mean_wer  mean_cer
adaptive       0  0.539796  0.299562
adaptive      10  0.317014  0.129862
adaptive      20  0.233458  0.072401
enhanced       0  0.609493  0.342590
enhanced      10  0.353730  0.148602
enhanced      20  0.270117  0.091941
     raw       0  0.319334  0.140451
     raw      10  0.189037  0.054233
     raw      20  0.183287  0.048989
```

## Subtitle Metrics
```json
{
  "reference_cues": 6,
  "predicted_cues": 10,
  "timestamp_mae_sec": 5.000000000000001,
  "reference_cps_violations": 0,
  "predicted_cps_violations": 3
}
```
