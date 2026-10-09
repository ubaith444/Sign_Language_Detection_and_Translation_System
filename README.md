# Indian Sign Language (ISL) Real-Time Detection, Recognition, and Translation System

---

## 1. WHAT: System Definition, Objectives, and Capabilities

### 1.1 Project Overview
The Indian Sign Language (ISL) Detection, Recognition, and Translation System is an enterprise-grade, edge-optimized artificial intelligence platform engineered to bridge communication barriers between Deaf and hard-of-hearing individuals and hearing non-signers. Operating on standard commodity CPU hardware without requiring dedicated GPUs or cloud connectivity, the system captures live monocular camera streams at 30+ frames per second (FPS), tracks 3D skeletal geometry across both hands, upper-body pose, and facial expressions, disambiguates subtle visual handshapes, isolates continuous gestures through kinetic velocity segmentation, and translates visual sign glosses into fluent, grammatically sound natural English with offline auditory speech synthesis.

The system is designed with a strict scientific distinction between **sign gesture classification** and **natural language translation**, acknowledging that recognizing isolated visual handshapes is fundamentally distinct from translating continuous visual-spatial grammar.

---

### 1.2 Core System Capabilities
1. **Holistic Real-Time Vision Tracking**: Continuously extracts 3D coordinates for 21 left hand joints, 21 right hand joints, 12 upper-body pose joints, and 20 facial non-manual keypoints, assembling a normalized 222-dimensional feature vector per frame.
2. **Decoupled Dual-Tier Machine Learning Engine**:
   * **Instantaneous Static Tier**: Evaluates static manual alphabet configurations (A through Z) and fixed manual postures in under 1.5 ms per frame with probability-calibrated unknown rejection.
   * **Temporal Dynamic Tier**: Evaluates continuous trajectory sequences (e.g., HELLO, PLEASE, THANK YOU, EMERGENCY) over a 30-frame sliding window using Bidirectional Gated Recurrent Units (BiGRU) with temporal self-attention in 8 to 14 ms.
3. **Continuous Kinetic Auto-Segmentation**: Employs per-landmark Euclidean displacement velocity and refractory debouncing to isolate discrete sign boundaries in continuous signing without requiring the signer to pause unnaturally.
4. **Non-Manual Facial Grammatical Modality Classifier**: Evaluates eyebrow elevation, furrowing, and head yaw oscillation to classify sentence grammatical moods (Declarative Statement, Polar Yes/No Question, Wh-Question, Negation).
5. **Phonological Sub-Unit Feature Extractor (HamNoSys)**: Decomposes 3D coordinates into Hamburg Notation System cheremes (Handshape, Palm Orientation, Place of Articulation, Movement, and Bimanual Symmetry) for linguistic interpretability.
6. **Syntactic ISL-to-English Natural Language Translation**: Restructures Subject-Object-Verb (SOV) and Topic-Comment gloss sequences into fluent Subject-Verb-Object (SVO) natural English with copula restoration, morphological inflection, and pronoun resolution.
7. **Two-Way Bidirectional Dialogue & Animated Sign Player**: Facilitates bidirectional conversation by converting English speech or text into ISL glosses and rendering real-time animated skeletal sign playback on an interactive HTML5 canvas.
8. **Offline Text-to-Speech (TTS) Voice Synthesis**: Converts translated English sentences into vocalized speech through the Windows SAPI5 audio pipeline with zero cloud dependencies or external API latency.
9. **Real-Time Live Webcam Calibration Engine**: Enables signers to adapt the active classification model to their specific hand proportions, camera distance, and room lighting in under 2 seconds directly from the live video stream.
10. **External Alphabet Dataset Ingestion Engine**: Ingests Kaggle image subfolder hierarchies (`A/*.jpg`, `B/*.jpg`) and CSV coordinate dumps, extracting 222-D coordinates via MediaPipe and retraining models with automated validation.
11. **Offline Pre-Recorded Video Batch Processor & Subtitle Burner**: Ingests recorded video files (.mp4, .avi, .mov), generates frame-accurate SubRip (.srt), WebVTT (.vtt), and JSON transcripts, and burns on-screen detection banners and subtitles directly into exported video containers.
12. **Session History, Dialogue Turn Logging & Multi-Format Exporter**: Chronologically tracks bidirectional conversation turns and exports full transcripts in formatted plain text (.txt), structured JSON (.json), and vector PDF (.pdf).
13. **Dual Interactive Interfaces**: Provides an enterprise web dashboard following strict light-mode design standards, alongside a native OpenCV desktop application with an on-screen heads-up display (HUD).

---

## 2. WHY: Linguistic, Architectural, and Engineering Rationale

### 2.1 The Linguistic Distinction: Sign Recognition vs. True Language Translation
A fundamental error in early gesture recognition systems is treating sign language as an encoded substitution cipher for spoken words. Indian Sign Language is a natural, visual-spatial language with its own independent phonology, morphology, and syntax:
* **Syntax Inversion (SOV vs. SVO)**: Natural English follows Subject-Verb-Object word order ("I am meeting the teacher"). ISL follows Subject-Object-Verb / Topic-Comment order (`ME TEACHER MEET`). Direct string concatenation produces ungrammatical, broken output (`I teacher meet`).
* **Copula Omission**: ISL omits linking verbs (copulas such as *is*, *am*, *are*, *was*, *were*). A translation engine must infer and restore the appropriate copula based on grammatical person, number, and tense.
* **Non-Manual Grammatical Markers (NMMs)**: In ISL, sentence modality (whether an utterance is a statement, a polar question, or a negative clause) is encoded through facial expressions and head gestures. Without facial modality analysis, a question cannot be distinguished from a statement.

---

### 2.2 Why Many Signs Look Identical (The Handshape Disambiguation Problem)
In sign language, many gestures form phonological minimal pairs that share near-identical visual silhouettes:
* **The Two-Handed ISL Vowel System (A, E, I, O, U)**: All five vowels utilize the exact same two-handed base posture: a non-dominant flat open palm facing the signer, and a dominant index finger pointing at it. The sole distinction is a 1.5 cm shift in contact point across the five fingertips (Thumb for A, Index for E, Middle for I, Ring for O, Pinky for U).
* **Closed-Fist Ambiguities (M, N, T, S, A, E)**: In single-handed signs, letters M, N, and T are all closed fists differing solely by whether the thumb is tucked under 3, 2, or 1 finger.
* **Monocular Perspective Flattening**: Flat 2D webcam sensors collapse depth, causing fingers to visually blend with the palm.
* **How This System Resolves Overlap**: By extracting 3D normalized coordinates, computing inter-joint finger curl angles, evaluating fingertip-to-thumb pairwise Euclidean distances, and tracking bimanual contact vectors, the system separates subtle minimal pairs that cause raw pixel classifiers to fail.

---

### 2.3 Signer-Independent Disjoint Splits (Eliminating Data Leakage)
Many published sign language models report inflated accuracy (>98%) because training and testing sets contain clips of the same signers in the same rooms with identical clothing. When deployed to a new user, accuracy drops drastically. This system enforces strict Leave-One-Signer-Out (LOSO) disjoint partitioning, isolating training clips from test clips by `signer_id` to ensure real-world generalization across unseen signers.

---

### 2.4 Edge Computing, User Privacy, and Zero Cloud Dependencies
Sign language interaction involves camera streams of users in private homes, classrooms, or workplaces. Streaming uncompressed video to cloud APIs introduces significant latency (>200 ms), high recurring costs, and severe privacy vulnerabilities. This system executes entirely on local CPU hardware: video frames remain in volatile local memory, background environmental pixels are masked, and speech is synthesized through local OS drivers.

---

## 3. HOW: End-to-End System Architecture and Workflow

```text
+----------------------------------------------------------------------------------------------------+
|                                    END-TO-END PIPELINE ARCHITECTURE                                |
+----------------------------------------------------------------------------------------------------+

  [Webcam Stream (DirectShow, 30 FPS)] OR [Offline Video Batch Container (.mp4, .avi)]
                            |
                            v
  [LandmarkExtractor (MediaPipe Tasks 0.10+ / Dual-Space Adaptive OpenCV Fallback)]
   - 21 Left Hand Joints (x, y, z)
   - 21 Right Hand Joints (x, y, z)
   - 12 Upper-Body Pose Joints (x, y, z)
   - 20 Facial Keypoints (x, y, z)
                            |
                            v
  [GeometricNormalizer (Scale, Position, and Invariance Engine)]
   - Wrist anchored at origin (0, 0, 0)
   - Hand scale normalized by wrist-to-middle MCP distance
   - Pose normalized by inter-shoulder span
   - Facial contour normalized by bounding span
   - Yields standardized 222-dimensional 1D coordinate vector
                            |
           +----------------+----------------+
           |                                 |
           v                                 v
  [Instantaneous Static Tier]       [Temporal Dynamic Tier]
   - Calibrated Random Forest        - 30-Frame Sliding Window (30 x 222)
   - Execution Time: < 1.5 ms        - 2-Layer BiGRU + Temporal Attention
   - Evaluates: Alphabet (A-Z)       - Execution Time: 8 - 14 ms
   - Few-shot retrainable (< 2s)     - Evaluates: Movement signs & phrases
           |                                 |
           +----------------+----------------+
                            |
                            v
  [Continuous Kinetic Segmenter]
   - Per-landmark Euclidean displacement velocity
   - Low-velocity transition rest phase detector
   - Refractory debouncing gate (suppresses duplicate emissions)
                            |
                            v
  [Non-Manual Facial Expression Classifier] + [HamNoSys Phonological Decomposer]
   - Eyebrow elevation & furrowing delta       - Handshape chereme categorization
   - Head yaw oscillation (negation)          - Palm orientation & articulation site
   - Modality: Statement / Question / Negation- Kinetic displacement classification
                            |
                            v
  [Syntactic ISL-to-English Translation Engine]
   - SOV to SVO syntactic restructuring
   - Copula restoration (am, is, are, was, were)
   - Morphological suffix agreement (-ing, -ed, -s)
   - Punctuation alignment based on facial modality cues
                            |
           +----------------+----------------+
           |                                 |
           v                                 v
  [Speech Synthesis Engine]         [Web / Desktop UI & Video Subtitler]
   - Windows SAPI5 / Web Speech      - Real-Time Video HUD & Landmark Canvas
   - Instant auditory vocalization   - Dialogue Studio & Sentence Builder
   - Zero cloud API latency          - SRT, WebVTT, JSON, and PDF Session Export
```

---

### 3.1 Step-by-Step Data Flow
1. **Frame Capture**: [camera_stream.py](file:///d:/Sign%20Language/src/vision/camera_stream.py) acquires frames via Windows DirectShow (`cv2.CAP_DSHOW`) at 30 FPS, providing resilient drop tolerance and background auto-reconnection.
2. **Landmark Extraction**: [landmark_extractor.py](file:///d:/Sign%20Language/src/vision/landmark_extractor.py) executes Google MediaPipe Tasks vision models ([hand_landmarker.task](file:///d:/Sign%20Language/data/hand_landmarker.task), [pose_landmarker_lite.task](file:///d:/Sign%20Language/data/pose_landmarker_lite.task), [face_landmarker.task](file:///d:/Sign%20Language/data/face_landmarker.task)).
3. **Geometric Normalization**: [normalizer.py](file:///d:/Sign%20Language/src/vision/normalizer.py) anchors coordinates to the anatomical root joint and scales distances by the palm baseline, outputting a 222-element floating-point array.
4. **Classification Dispatch**:
   * Static handshapes and alphabet letters are evaluated in **1.2 to 1.5 ms** via [static_classifier.py](file:///d:/Sign%20Language/src/models/static_classifier.py).
   * Continuous multi-frame signs are evaluated over 30 buffered frames in **8 to 14 ms** via [dynamic_classifier.py](file:///d:/Sign%20Language/src/models/dynamic_classifier.py).
5. **Continuous Segmentation**: [continuous_segmenter.py](file:///d:/Sign%20Language/src/models/continuous_segmenter.py) detects motion onset and rest phases to commit signs without manual triggers.
6. **Linguistic Refinement**: [facial_modality_classifier.py](file:///d:/Sign%20Language/src/vision/facial_modality_classifier.py) and [hamnosys_extractor.py](file:///d:/Sign%20Language/src/linguistics/hamnosys_extractor.py) assign grammatical mood and phonetic properties.
7. **Syntactic Translation**: [sentence_builder.py](file:///d:/Sign%20Language/src/translation/sentence_builder.py) and [english_to_isl.py](file:///d:/Sign%20Language/src/translation/english_to_isl.py) restructure gloss tokens into fluent natural language sentences.
8. **Audio and Subtitle Output**: Translated sentences are spoken aloud via [speech_synthesizer.py](file:///d:/Sign%20Language/src/audio/speech_synthesizer.py) and serialized to standard subtitle files.

---

### 3.2 Real-Time Live Webcam Calibration Engine (Few-Shot Fine-Tuning)
To overcome individual differences in hand proportions, camera distance, and room lighting, the system includes an in-memory few-shot calibration engine ([realtime_finetuner.py](file:///d:/Sign%20Language/src/models/realtime_finetuner.py)):
1. **User Sample Accumulation**: The user selects a target sign (e.g. `Letter A`) and clicks **Capture Frame** on the web dashboard to collect 5 to 10 live webcam landmark vectors.
2. **Geometric Perturbation Augmentation**: For each captured vector, the engine generates 5 augmented variations by applying:
   * Isotropic scaling ($\pm 3\%$)
   * Planar translation ($\pm 0.015$ normalized units)
   * Gaussian coordinate jitter ($\sigma = 0.008$)
3. **In-Memory Incremental Retraining**: The augmented user pool is merged with baseline synthetic prototypes and retrained via Scikit-Learn in **1.57 seconds** on CPU.
4. **Instant Hot-Reload**: The updated decision boundaries take effect immediately without restarting the web server or terminating the live camera stream.

---

### 3.3 External Alphabet Dataset Ingestion Engine
To expand vocabulary and incorporate large public datasets (e.g., Kaggle ISL alphabet datasets), the system provides an automated ingestion engine ([dataset_importer.py](file:///d:/Sign%20Language/scripts/dataset_importer.py)):
1. **Folder Hierarchy Scanning**: Scans image folders formatted as `A/*.jpg`, `B/*.jpg`, etc., or tabular CSV landmark files.
2. **Automated Landmark Extraction**: Processes raw images through MediaPipe Tasks Vision, discarding invalid or undetectable frames.
3. **Standardized Archiving**: Packs extracted coordinates and class labels into compressed `.npz` archives inside `data/imported_datasets/`.
4. **Automated Classifier Training**: Automatically retrains the static classifier, evaluates accuracy on a held-out test split, and updates model weights in `data/trained_models/`.

---

### 3.4 Two-Way Communication & Skeletal Sign Player
Bidirectional communication allows hearing non-signers to communicate back to Deaf signers:
1. **Input Ingestion**: The hearing user speaks via browser microphone (Web Speech API) or types an English sentence.
2. **Grammar Transformation**: [english_to_isl.py](file:///d:/Sign%20Language/src/translation/english_to_isl.py) converts English Subject-Verb-Object syntax into ISL Subject-Object-Verb gloss order, removing copulas and tagging question markers.
3. **Keyframe Synthesis**: [sign_player_data.py](file:///d:/Sign%20Language/src/translation/sign_player_data.py) retrieves anatomical 3D skeletal keyframes for each gloss.
4. **Canvas Rendering**: An interactive HTML5 canvas renders smooth skeletal joint animations at selectable playback speeds (0.5x, 1.0x, 1.5x) with play, pause, step, and loop controls.

---

### 3.5 Pre-Recorded Video Batch Processor & Subtitle Burner
Offline video files are processed via [video_batch_processor.py](file:///d:/Sign%20Language/src/vision/video_batch_processor.py):
1. **Container Decoding**: Reads `.mp4`, `.avi`, or `.mov` files frame-by-frame via OpenCV.
2. **Temporal Alignment**: Continuously extracts landmarks, classifies signs, segments transitions, and computes precise millisecond timestamps.
3. **Standardized Subtitle Export**: Generates frame-accurate SubRip (`.srt`), WebVTT (`.vtt`), and structured JSON transcript files.
4. **On-Screen Subtitle Burn-In**: Renders high-contrast bounding boxes, landmark skeletons, and translated text banners directly onto the video frames, saving a fully subtitled output video.

---

### 3.6 Session Management & Multi-Format Audit Logging
Dialogue sessions are recorded in chronological turns via [session_manager.py](file:///d:/Sign%20Language/src/session/session_manager.py):
* **Turn Logging**: Tracks speaker role, input text, translated text, grammatical modality, and confidence score.
* **Structured JSON Export**: Complete session state formatted for database ingestion.
* **Plain Text Transcript**: Clean, readable conversational transcript.
* **Vector PDF Export**: Clean, multi-column PDF generated via vector graphics with session statistics, speaker badges, and dialogue chronology.

---

## 4. DETAILED TECH STACK AND MODEL SELECTION

### 4.1 Technology Stack Breakdown

| Category | Technology | Version | Purpose (What) | Architectural Rationale (Why) |
| :--- | :--- | :--- | :--- | :--- |
| **Language & Runtime** | Python | 3.11.x | Core execution runtime for computer vision, machine learning, linguistics, and server APIs. | Provides optimal ecosystem compatibility for MediaPipe, OpenCV, PyTorch, Scikit-Learn, and native C-extension audio drivers with stable Windows execution. |
| **Vision & Keypoint Tracking** | MediaPipe Tasks & OpenCV | 0.10.x / 4.11+ | Real-time extraction of 3D landmarks for hands (42 pts), upper-body pose (12 pts), and facial cues (20 pts). | Tasks API delivers robust 3D skeletal tracking under all lighting and skin tones with CPU inference in < 16 ms. Dual-space OpenCV fallback ensures zero-crash operation. |
| **Video I/O & Frame Processing** | OpenCV (cv2) | 4.11+ | DirectShow hardware webcam acquisition, video container decoding, frame annotation, and MJPEG encoding. | Industry standard providing low-latency hardware capture (`CAP_DSHOW`), high-throughput frame conversion, and native subtitle burn-in. |
| **Numerical Processing** | NumPy | 1.26.4 | Multi-dimensional array operations, feature normalization, Euclidean distance math, and sliding buffers. | Optimized C-level array math. Pinned to `numpy<2` to guarantee strict ABI binary compatibility with MediaPipe and OpenCV modules. |
| **Machine Learning (Static & Alphabet)** | Scikit-Learn | 1.6+ | Classification of instantaneous manual configurations and fingerspelling (A-Z) using Random Forest and MLP. | Sub-millisecond inference (< 1.5 ms), robust orthogonal decision boundaries on joint distances, and sub-2-second on-device fine-tuning. |
| **Deep Learning (Dynamic Sequences)** | PyTorch | 2.6+ | Temporal sequence modeling via Bidirectional GRU with self-attention for continuous multi-frame signs. | Handles variable-length temporal sequences, captures bidirectional motion onset/decay, and executes in < 12 ms on CPU. |
| **Facial Modality Classification** | Geometric Head Pose & Brow Engine | In-house | Grammatical sentence modality classification (Statement, Polar Question, Wh-Question, Negation). | Non-manual markers carry essential grammatical functions in ISL; refines syntax and punctuation without relying solely on manual gestures. |
| **Phonological Analysis** | HamNoSys Decomposition Engine | In-house | Sub-lexical chereme extraction (Handshape, Orientation, Location, Movement, Relation). | Bridges continuous numerical coordinates with formal sign language phonetics, enabling phonetic transcriptions and linguistic search. |
| **Syntactic Translation** | Neural & Rule-Based Translator | In-house | Converts ISL SOV gloss sequences into fluent English SVO sentences with copula and article insertion. | Addresses fundamental syntax divergence: direct gloss concatenation produces broken English; dedicated translator restores grammar. |
| **Speech Synthesis (TTS)** | pyttsx3 & Web Speech API | 2.99 / HTML5 | Dual-mode text-to-speech output using Windows SAPI5 offline and browser-native speech synthesis. | Complete user privacy, zero cloud dependency, zero external API billing, and instantaneous local auditory vocalization. |
| **Web Server & APIs** | Flask & Flask-CORS | 3.1+ | RESTful endpoints, live MJPEG streaming, real-time calibration APIs, and transcript export handlers. | Minimal runtime footprint, low overhead, seamless OpenCV multipart frame streaming, and clean asynchronous client polling. |
| **Frontend Architecture** | Semantic HTML5, Vanilla CSS3, ES6+ JS | Modern Web Standards | Accessible dashboard providing live video HUD, sentence studio, calibration panel, and evaluation charts. | Adheres to strict enterprise light-mode standards, eliminating heavy npm bundles while delivering microsecond responsiveness. |
| **Real-Time Calibration Engine** | Few-Shot Incremental Fine-Tuner | In-house / sklearn | In-memory adaptation directly from live webcam stream for user-specific hand geometry. | Resolves inter-class handshape minimal pairs by fine-tuning decision boundaries in < 2 seconds on standard CPU hardware. |
| **Alphabet Dataset Importer** | Batch Ingestion CLI & API | In-house | Automated directory scanning, landmark extraction, and model training for external ISL datasets. | Enables seamless ingestion of Kaggle ISL alphabet folders or coordinate CSV dumps into compressed archives. |

---

### 4.2 Architectural Trade-Off Analysis

#### Vision: MediaPipe Tasks vs. OpenPose vs. YOLO-Pose
* **OpenPose**: Highly accurate full-body keypoint detection, but requires dedicated GPU hardware and runs at < 4 FPS on commodity CPUs.
* **YOLO-Pose**: Excellent for person detection and major body joints, but lacks granular 21-joint 3D hand topology required to disambiguate fingertip minimal pairs.
* **Selected (MediaPipe Tasks Vision)**: Executes in 12 to 16 ms per frame on CPU, isolates 42 hand joints in 3D metric coordinates, and provides lightweight single-binary `.task` model bundles.

#### Static Classification: Random Forest vs. End-to-End Pixel CNN
* **End-to-End CNN (ResNet, MobileNet on raw images)**: Requires millions of parameters, sensitive to room background, lighting, and skin pigmentation, with CPU inference times of 35 to 60 ms per frame.
* **Selected (Random Forest on 222-D Normalized Landmarks)**: Evaluates 80 orthogonal decision trees in < 1.5 ms on CPU, impervious to background colors and lighting, and retrainable in 1.57 seconds for live webcam calibration.

#### Video Acquisition: DirectShow (`cv2.CAP_DSHOW`) vs. Media Foundation (MSMF)
* **Default MSMF**: Standard OpenCV Windows backend often suffers from 5 to 10 second warm-up latencies, dropped frames during initialization, and format negotiation stalls.
* **Selected (DirectShow)**: Delivers instant hardware connection, stable 30 FPS throughput, and robust drop tolerance.

#### Audio Vocalization: Local pyttsx3 (SAPI5) vs. Cloud Speech APIs
* **Cloud APIs (Google Cloud Text-to-Speech, Amazon Polly)**: Introduce 150 to 400 ms round-trip network latency, require API authentication keys, incur per-character billing, and transmit user conversation over the Internet.
* **Selected (pyttsx3 / Windows SAPI5)**: Zero latency overhead, completely offline, zero operating cost, and 100% private.

---

### 4.3 Comprehensive Model Selection Analysis

```text
+-----------------------------------------------------------------------------------------------------------------------------+
|                                              CANDIDATE MODEL COMPARISON MATRIX                                              |
+-----------------------------------------------------------------------------------------------------------------------------+
| Candidate Architecture         | Handshape Detail | Temporal Dynamics | CPU Latency | Retrain Time | Parameter Size | Status   |
|--------------------------------|------------------|-------------------|-------------|--------------|----------------|----------|
| End-to-End Video Transformer   | High             | Very High         | > 140 ms    | Hours (GPU)  | 85M - 300M     | REJECTED |
| 3D-CNN (I3D / ResNet-3D)       | Moderate         | High              | 65 - 95 ms  | Hours (GPU)  | 25M - 48M      | REJECTED |
| Spatial-Temporal GCN (ST-GCN)  | Very High        | Very High         | 24 - 40 ms  | Minutes (GPU)| 3.2M           | RESERVED |
| 1D Temporal CNN (TCN)          | Moderate         | Moderate          | 4 - 8 ms    | ~ 15 seconds | 0.8M           | BACKUP   |
| Multi-Layer Perceptron (MLP)   | High             | None              | < 1.0 ms    | ~ 2.5 seconds| 0.4M           | BACKUP   |
| Random Forest (Landmarks)      | High             | None              | < 1.5 ms    | 1.57 seconds | 3.3 MB         | SELECTED |
| BiGRU + Attention (Landmarks)  | Moderate         | Exceptional       | 8 - 14 ms   | ~ 30 seconds | 0.78 MB        | SELECTED |
+-----------------------------------------------------------------------------------------------------------------------------+
```

#### Why Raw Pixel Vision Models Were Eliminated
End-to-end pixel networks (such as 3D-CNNs and Video Transformers) process $640 \times 480 \times 3 \times 30 \approx 27.6\text{ million}$ numerical inputs per second. On CPU hardware, inference takes over 100 ms per frame, falling far short of the 30 FPS real-time threshold. Furthermore, they overfit to room lighting, background colors, and skin tones, failing when evaluated on unseen signers.

#### Why the Decoupled Hybrid Architecture Was Selected
1. **Decoupled Landmark Representation**: MediaPipe Tasks reduces each image tensor to 222 normalized coordinates, abstracting away background and lighting variations.
2. **Calibrated Random Forest for Instantaneous Signs**: Random Forest operates on single-frame geometric coordinates, evaluating 80 decision trees in **< 1.5 ms** with native posterior probability calibration for unknown rejection. It retrains in **1.57 seconds** on CPU, enabling instant webcam calibration.
3. **BiGRU with Attention for Continuous Sequences**: Gated Recurrent Units model bidirectional temporal dependencies across 30 frames with half the parameter count of LSTMs. Temporal attention weights the peak inflection frames of a gesture, completing inference in **8 to 14 ms** on CPU.

---

## 5. WHERE: Codebase Directory and File Mapping

Below is the file-by-file directory map of the codebase:

```text
d:/Sign Language/
├── data/                                      <- Model assets, lexicon definitions, and dataset specifications
│   ├── hand_landmarker.task                   <- Google MediaPipe Tasks 21-joint 3D hand tracking model
│   ├── pose_landmarker_lite.task              <- Google MediaPipe Tasks upper-body pose tracking model
│   ├── face_landmarker.task                   <- Google MediaPipe Tasks facial mesh tracking model
│   ├── isl_alphabet.json                      <- ISL fingerspelling dataset specification (A-Z)
│   ├── isl_continuous_fingerspelling.json     <- Aligned continuous fingerspelling benchmark specification
│   ├── isl_lexicon.json                       <- Core vocabulary definitions and phonological parameters
│   ├── parallel_corpus.json                   <- ISL Subject-Object-Verb to English SVO parallel corpus
│   ├── evaluation_report.json                 <- Benchmark evaluation metrics and confusion matrices
│   ├── signer_independent_report.json         <- Leave-One-Signer-Out (LOSO) evaluation metrics
│   ├── sample_benchmark_isl.mp4               <- Pre-recorded benchmark video clip for offline validation
│   ├── imported_datasets/                     <- Storage directory for external ingested datasets (.npz)
│   ├── processed_videos/                      <- Storage for batch-processed videos, subtitles, and transcripts
│   ├── raw_recordings/                        <- Storage for recorded consented training clips
│   ├── transcripts/                           <- Exported session history transcripts (PDF, TXT, JSON)
│   ├── trained_models/                        <- Serialized machine learning model weights
│   │   ├── isl_static_model.joblib            <- Trained calibrated static Random Forest model
│   │   ├── isl_dynamic_gru.pt                 <- Trained PyTorch BiGRU model for dynamic gestures
│   │   └── dynamic_metadata.json              <- Metadata specification for dynamic classes
│   └── user_calibration/                     <- Persisted few-shot webcam calibration samples (.npz)
├── scripts/                                   <- Command-line utilities and execution entrypoints
│   ├── run_web_app.py                         <- Launches the interactive Flask web dashboard
│   ├── run_desktop_app.py                     <- Launches native OpenCV window with on-screen HUD
│   ├── dataset_importer.py                    <- External dataset ingestion CLI (Kaggle folders / CSVs)
│   ├── evaluate_system.py                     <- Automated quantitative benchmark evaluation runner
│   ├── process_video.py                       <- Offline video batch processor and subtitle burner
│   ├── record_samples.py                      <- Consented ISL dataset recorder with metadata logging
│   └── train_models.py                        <- Model training pipeline script
├── src/                                       <- Core application source code
│   ├── config.py                              <- Global paths, thresholds, and dimension constants
│   ├── audio/                                 <- Audio synthesis and voice drivers
│   │   └── speech_synthesizer.py              <- Offline Windows SAPI5 text-to-speech engine
│   ├── evaluation/                            <- Empirical metrics and benchmarking suites
│   │   ├── metrics.py                         <- Accuracy, F1-score, WER, and latency profiler
│   │   └── test_benchmark.py                  <- Automated evaluation benchmark suite
│   ├── linguistics/                           <- Linguistic decomposition and phonetics
│   │   └── hamnosys_extractor.py              <- HamNoSys phonological feature decomposition engine
│   ├── models/                                <- Machine learning classification models
│   │   ├── static_classifier.py               <- Instantaneous Random Forest / MLP static classifier
│   │   ├── dynamic_classifier.py              <- PyTorch BiGRU dynamic sequence classifier
│   │   ├── continuous_segmenter.py            <- Kinetic energy velocity and debounce segmenter
│   │   ├── realtime_finetuner.py              <- Live webcam few-shot calibration engine
│   │   └── signer_independent_eval.py         <- Leave-One-Signer-Out validation engine
│   ├── session/                               <- Dialogue session logging and export
│   │   └── session_manager.py                 <- Turn tracking and PDF/TXT/JSON transcript exporter
│   ├── translation/                           <- Grammar transformation and sign player
│   │   ├── english_to_isl.py                  <- English-to-ISL grammar transformer (SVO to SOV)
│   │   ├── grammar_rules.py                   <- Linguistic inflection rules and copula restoration
│   │   ├── sentence_builder.py                <- Real-time gloss accumulator and translation studio
│   │   ├── seq2seq_translator.py              <- Neural and rule-based gloss-to-English translator
│   │   └── sign_player_data.py                <- Skeletal animation keyframes for 2-way sign player
│   ├── vision/                                <- Video capture and landmark extraction
│   │   ├── camera_stream.py                   <- Threaded DirectShow webcam capture with auto-reconnect
│   │   ├── landmark_extractor.py              <- MediaPipe Tasks vision pipeline & OpenCV tracker
│   │   ├── normalizer.py                      <- Scale-invariant geometric normalizer (222-D vector)
│   │   ├── facial_modality_classifier.py      <- Non-manual facial grammatical expression classifier
│   │   └── video_batch_processor.py           <- Offline video file batch processor and subtitle burner
│   └── web/                                   <- Web server and interactive dashboard
│       ├── app.py                             <- Flask backend application and streaming endpoints
│       ├── static/                            <- Frontend styles and client-side logic
│       │   ├── css/style.css                  <- Strict light-mode design system styles
│       │   └── js/app.js                      <- Real-time client controller and telemetry polling
│       └── templates/                         <- HTML dashboard templates
│           └── index.html                     <- Main interactive application dashboard
├── tests/                                     <- Automated test suites (100% passing)
│   ├── test_advanced_features.py              <- Tests HamNoSys, facial modality, 2-way player, session
│   ├── test_batch_processor.py                <- Tests offline video batch processing & subtitle exports
│   └── test_finetuning_and_importer.py        <- Tests live few-shot calibration and dataset ingestion
├── requirements.txt                           <- Pinned project dependencies
└── TECH_STACK.md                              <- Detailed technology stack and rationale reference
```

---

## 6. Installation, Configuration, and Operation

### 6.1 Prerequisites
* Windows 10 or Windows 11 (64-bit)
* Python 3.11.x (Installed and available in PATH)
* Standard USB webcam or integrated laptop camera

### 6.2 Setup Environment
```powershell
# Navigate to repository:
cd "d:\Sign Language"

# Create virtual environment:
python -m venv .venv

# Activate virtual environment:
.venv\Scripts\Activate.ps1

# Install pinned dependencies:
pip install -r requirements.txt
```

---

### 6.3 Launching Applications

#### 1. Interactive Web Interface (Primary Application)
```powershell
.venv\Scripts\python.exe scripts/run_web_app.py
```
Open your browser to: **`http://127.0.0.1:5000/`**

The web dashboard provides:
* **Live Video HUD**: Displays real-time webcam feed with skeletal overlays, detection badges, and confidence metrics.
* **Dialogue Studio**: Real-time sentence accumulator with audio synthesis, undo, clear, and copy controls.
* **Two-Way Sign Player**: Interactive animated sign visualizer with variable playback speed controls.
* **Video Batch Processing Dropzone**: Upload pre-recorded video files to generate SRT, WebVTT, and burned-in subtitled clips.
* **Controls & Few-Shot Calibration Tab**: Capture webcam frames for specific signs and retrain the model in under 2 seconds.
* **System Evaluation Tab**: View live confusion matrices, latency telemetry, and model performance metrics.

---

#### 2. Native Desktop OpenCV Interface
```powershell
.venv\Scripts\python.exe scripts/run_desktop_app.py
```
* **Keyboard Shortcuts**:
  * `[A]` : Add detected sign to sentence
  * `[U]` : Undo last word
  * `[C]` : Clear sentence
  * `[S]` : Speak sentence aloud via TTS
  * `[P]` : Toggle privacy background blur
  * `[M]` : Switch mode (Static Signs <-> Dynamic Sequences)
  * `[Q]` : Quit desktop application

---

#### 3. Real-Time Webcam Calibration (Few-Shot Fine-Tuning)
To calibrate the model to your specific hand geometry directly from the web interface:
1. Open the **Controls** tab on `http://127.0.0.1:5000/?tab=settings-panel`.
2. Select your target sign from the dropdown (e.g. `Letter A` or `Letter M`).
3. Position your hand clearly in front of the camera and click **Capture Frame** (capture 5 to 10 frames).
4. Click **Fine-Tune Model Now (1-Click)**. The model will adapt in **1.57 seconds** without interrupting the stream.

---

#### 4. Ingesting External Alphabet Datasets
```powershell
# Ingest Kaggle ISL alphabet image folders (A/, B/, ...) and retrain:
.venv\Scripts\python.exe scripts/dataset_importer.py --input-dir "D:/path/to/isl_alphabet_dataset" --train

# Ingest pre-extracted landmark coordinates from a CSV file:
.venv\Scripts\python.exe scripts/dataset_importer.py --csv "D:/path/to/landmarks.csv" --train
```

---

#### 5. Offline Video Batch Subtitle Generation
```powershell
# Process an offline video file and generate SRT, WebVTT, JSON, and subtitled MP4:
.venv\Scripts\python.exe scripts/process_video.py "data/sample_benchmark_isl.mp4" --burn-subtitles
```

---

#### 6. Executing Automated Benchmark Evaluation
```powershell
.venv\Scripts\python.exe scripts/evaluate_system.py
```
Generates comprehensive quantitative evaluation metrics, macro F1-scores, and confusion matrix reports saved to [data/evaluation_report.json](file:///d:/Sign%20Language/data/evaluation_report.json).

---

#### 7. Executing Unit and Integration Test Suite
```powershell
.venv\Scripts\python.exe -m unittest discover tests
# 21 tests ran - OK (0 failures, 0 errors)
```
