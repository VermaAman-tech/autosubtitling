
# EE 679 — Full Evaluation Report (medium.en)

**Trailers evaluated**: 14

**Model**: medium.en

**Mean WER**: 0.5224   **Mean CER**: 0.4274   **Mean chrF**: 55.5   **Mean RTF**: 0.1746


## Per-Trailer Results

| slug | genre | duration_sec | pred_cues | ref_cues | wer | cer | chrf | bleu1 | rtf | cps_violations | scene_dominant_mood | scene_music_fraction |

|---|---|---|---|---|---|---|---|---|---|---|---|---|

| Marvel_s_The_Avengers__Trailer__OFFICIAL | action | 125.000 | 18 | 22 | 0.144 | 0.122 | 84.940 | 84.500 | 0.115 | 8 | epic | 0.056 |

| YTDown_YouTube_SPIDER_MAN_BRAND_NEW_DAY_ | unknown | 160.100 | 37 | 18 | 0.183 | 0.090 | 86.320 | 87.360 | 0.190 | 14 | epic | 0.000 |

| avengers_age_of_ultron | action | 145.100 | 2 | 25 | 0.948 | 0.942 | 6.020 | 0.000 | 0.108 | 0 | epic | 0.000 |

| bridesmaids | comedy | 130.100 | 35 | 31 | 0.449 | 0.325 | 61.330 | 64.330 | 0.254 | 12 | epic | 0.052 |

| captain_america_civil_war | action | 138.100 | 0 | 20 | 1.000 | 1.000 | 0.000 | 0.000 | 0.099 | 0 | epic | 0.029 |

| dunkirk | action | 144.100 | 9 | 20 | 0.513 | 0.392 | 58.780 | 49.830 | 0.052 | 0 | epic | 0.027 |

| mad_max_fury_road | action | 150.000 | 34 | 30 | 0.450 | 0.264 | 63.510 | 56.900 | 0.187 | 15 | epic | 0.021 |

| nomadland | drama | 153.100 | 69 | 45 | 0.533 | 0.393 | 63.730 | 62.730 | 0.404 | 37 | epic | 0.079 |

| spider_man_into_spider_verse | action | 143.300 | 12 | 24 | 0.570 | 0.514 | 48.800 | 36.370 | 0.081 | 0 | epic | 0.039 |

| the_dark_knight | action | 136.400 | 11 | 11 | 0.525 | 0.355 | 61.180 | 54.370 | 0.045 | 3 | epic | 0.000 |

| the_revenant | drama | 275.100 | 57 | 86 | 0.554 | 0.566 | 47.420 | 32.590 | 0.287 | 2 | epic | 0.116 |

| the_social_network | drama | 80.800 | 26 | 14 | 0.634 | 0.488 | 57.100 | 47.020 | 0.204 | 7 | dialogue | 0.010 |

| whiplash | drama | 186.600 | 42 | 38 | 0.405 | 0.266 | 68.930 | 67.030 | 0.201 | 19 | epic | 0.161 |

| zodiac | thriller | 186.600 | 42 | 38 | 0.405 | 0.266 | 68.930 | 67.030 | 0.217 | 19 | epic | 0.161 |


## Genre-Level Summary

| genre | WER | CER | chrF | RTF |

|---|---|---|---|---|

| action | 0.5929 | 0.5128 | 46.2 | 0.0981 |

| comedy | 0.4494 | 0.3251 | 61.3 | 0.2536 |

| drama | 0.5315 | 0.4283 | 59.3 | 0.2741 |

| thriller | 0.4050 | 0.2664 | 68.9 | 0.2169 |

| unknown | 0.1832 | 0.0897 | 86.3 | 0.1900 |


## Genre Classifier (MFCC k-NN LOO)

- Overall LOO accuracy: **77%**  (13 trailers)

| Genre | Accuracy |

|---|---|

| action | 100% |

| comedy | 0% |

| drama | 75% |

| thriller | 0% |


## Key Findings

**Best transcribed trailers** (lowest WER):

  - Marvel_s_The_Avengers__Trailer__OFFICIAL: WER=0.144, chrF=84.9 (action)

  - YTDown_YouTube_SPIDER_MAN_BRAND_NEW_DAY_: WER=0.183, chrF=86.3 (unknown)

  - whiplash: WER=0.405, chrF=68.9 (drama)


**Hardest to transcribe** (highest WER):

  - captain_america_civil_war: WER=1.000, chrF=0.0 (action)

  - avengers_age_of_ultron: WER=0.948, chrF=6.0 (action)

  - the_social_network: WER=0.634, chrF=57.1 (drama)


**Music fraction vs WER correlation**: r = -0.252  (negative — unexpected)
