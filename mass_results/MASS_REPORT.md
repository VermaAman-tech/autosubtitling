
# EE 679 — Mass Trailer Evaluation + Scene Understanding Report

Videos processed: **14**

Mean WER: **0.8615**  Mean CER: **0.7664**  Mean RTF: **0.0944**


## Per-Trailer Results

| slug | duration_sec | pred_cues | ref_cues | wer | cer | asr_rtf_overall | cps_violations |

|---|---|---|---|---|---|---|---|

| Marvel's The Avengers- Trailer (OFFICIAL) - Marvel Entertainment (1080p, h264) | 124.97 | 9 | 22 | 0.6829 | 0.5754 | 0.0381 | 2 |

| YTDown_YouTube_SPIDER-MAN-BRAND-NEW-DAY-Official-Traile_Media_8TZMtslA3UY_002_720p | 160.12 | 11 | 18 | 0.6584 | 0.5288 | 0.0423 | 3 |

| avengers_age_of_ultron | 145.14 | 19 | 53 | 0.8623 | 0.748 | 0.0917 | 7 |

| bridesmaids | 130.06 | 22 | 90 | 0.8921 | 0.7999 | 0.0782 | 10 |

| captain_america_civil_war | 138.12 | 8 | 44 | 0.95 | 0.8989 | 0.0712 | 2 |

| dunkirk | 144.13 | 6 | 47 | 0.9634 | 0.8934 | 0.0433 | 1 |

| mad_max_fury_road | 150.0 | 19 | 69 | 0.895 | 0.8164 | 0.0856 | 6 |

| nomadland | 153.05 | 34 | 102 | 0.9451 | 0.8528 | 0.2137 | 17 |

| spider_man_into_spider_verse | 143.3 | 17 | 41 | 0.7993 | 0.7164 | 0.0589 | 7 |

| the_dark_knight | 136.41 | 12 | 26 | 0.9328 | 0.8365 | 0.0715 | 0 |

| the_revenant | 275.06 | 62 | 190 | 0.8272 | 0.7532 | 0.1522 | 26 |

| the_social_network | 80.82 | 15 | 28 | 0.9323 | 0.7612 | 0.2293 | 3 |

| whiplash | 186.59 | 24 | 93 | 0.8603 | 0.7747 | 0.0745 | 5 |

| zodiac | 186.59 | 24 | 93 | 0.8603 | 0.7738 | 0.0712 | 5 |


## Scene Understanding Summary

| slug | scene_dominant_mood | scene_music_fraction | scene_speech_fraction | scene_n_segments | scene_n_boundaries |

|---|---|---|---|---|---|

| Marvel's The Avengers- Trailer (OFFICIAL) - Marvel Entertainment (1080p, h264) | epic | 0.056 | 0.296 | 37 | 37 |

| YTDown_YouTube_SPIDER-MAN-BRAND-NEW-DAY-Official-Traile_Media_8TZMtslA3UY_002_720p | epic | 0.0 | 0.324 | 47 | 47 |

| avengers_age_of_ultron | epic | 0.0 | 0.165 | 20 | 19 |

| bridesmaids | epic | 0.052 | 0.25 | 22 | 21 |

| captain_america_civil_war | epic | 0.029 | 0.065 | 15 | 14 |

| dunkirk | epic | 0.027 | 0.062 | 21 | 20 |

| mad_max_fury_road | epic | 0.021 | 0.337 | 41 | 40 |

| nomadland | epic | 0.079 | 0.528 | 83 | 84 |

| spider_man_into_spider_verse | epic | 0.039 | 0.017 | 9 | 8 |

| the_dark_knight | epic | 0.0 | 0.044 | 13 | 12 |

| the_revenant | epic | 0.116 | 0.584 | 140 | 144 |

| the_social_network | dialogue | 0.01 | 0.415 | 35 | 36 |

| whiplash | epic | 0.161 | 0.043 | 16 | 15 |

| zodiac | epic | 0.161 | 0.043 | 16 | 15 |


## Novel Experiment B — Noise-Type Classifier

Overall accuracy: **0.3156**

Per-class accuracy:

  - babble: 0.0000

  - music/soundtrack: 0.4000

  - pink_noise: 0.0000

  - silence: 1.0000

  - white_noise: 0.5200


## Novel Experiment D — Streaming Pipeline Latency

| Mode | First-Word Latency (ms) | WER | RTF |

|---|---|---|---|

| batch | 255 | 0.1744 | 0.0317 |

| stream_1.0s | 195 | 0.6066 | 0.2602 |

| stream_2.0s | 186 | 0.3241 | 0.2322 |

| stream_3.0s | 203 | 0.2907 | 0.0872 |

| stream_5.0s | 233 | 0.2619 | 0.1424 |
